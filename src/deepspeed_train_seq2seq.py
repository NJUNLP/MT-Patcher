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
    AutoTokenizer,
    Seq2SeqTrainer,
    M2M100ForConditionalGeneration,
    DataCollatorForSeq2Seq
    
)

from dataclasses import dataclass, field
from typing import Optional

from src.tasks.utils import RemoveDeepspeedCheckpointCallback
from src.data import make_data_module
from dataclasses import dataclass, field

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.

logger = get_logger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")

@dataclass
class DataArguments:
    train_file: str = field(default=None, metadata={"help": "Path to the training data."})
    validation_file: str = field(default=None, metadata={"help": "Path to the training data."})

    srclang: str = field(default="zh")
    tgtlang: str = field(default="en")

@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def get_model(model_args):

    model = M2M100ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,trust_remote_code=True,low_cpu_mem_usage=True,
    )

    return model

def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path,trust_remote_code=True,padding_side="right")

    model = get_model(model_args)

    train_dataset = make_data_module(data_args,model_args,tokenizer,type="seq2seq")
    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # Preprocessing the datasets.
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=[RemoveDeepspeedCheckpointCallback], 
        train_dataset=train_dataset,
        data_collator=data_collator)
    trainer.train()
    model.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()