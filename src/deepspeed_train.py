#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.


from accelerate.logging import get_logger
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    Trainer
)

from dataclasses import dataclass, field
from typing import Optional
import torch

from src.tasks.utils import RemoveDeepspeedCheckpointCallback
from src.data import make_data_module
from dataclasses import dataclass, field


from peft import get_peft_model, TaskType, LoraConfig
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.

logger = get_logger(__name__)


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    tokenizer_path: Optional[str] = field(default="facebook/opt-125m")
    use_peft: bool = field(
        default=False
    )
    # LoRA Args
    lora_r: Optional[int] = field(
        default=4
    )
    lora_alpha: Optional[int] = field(
        default=32
    )
    lora_dropout: Optional[float] = field(
        default=0.1
    )

@dataclass
class DataArguments:
    train_file: str = field(default=None, metadata={"help": "Path to the training data."})
    validation_file: str = field(default=None, metadata={"help": "Path to the training data."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def get_model(model_args):
    if "llama" in model_args.model_name_or_path:
        print("Using SDPA for Llama")
        from src.models.modeling_llama import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,trust_remote_code=True,low_cpu_mem_usage=True,
        )
        # st = time.time()
        # model = torch.compile(model)
        # print("Compiling model takes {}s".format(time.time() - st))
    else:
        config = AutoConfig.from_pretrained(model_args.tokenizer_path,trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path,config=config,low_cpu_mem_usage=True,trust_remote_code=True)
    if model_args.use_peft:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False, 
            r = model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout
            )
        model = get_peft_model(model,peft_config)
        model = model.half()
        model.print_trainable_parameters()

    return model

def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_path,trust_remote_code=True,padding_side="right",use_fast=False)

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    model = get_model(model_args)

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == 64000: # 64000 for baichuan model (older version)
        tokenizer.pad_token_id = 0 # set as the <unk> token
    data_module = make_data_module(data_args,model_args,tokenizer,type="sft")

    # Preprocessing the datasets.
    trainer = Trainer(model=model,tokenizer=tokenizer,args=training_args,callbacks=[RemoveDeepspeedCheckpointCallback], **data_module)
    trainer.train()
    model.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()