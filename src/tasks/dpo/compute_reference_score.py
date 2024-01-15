from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import transformers
import torch

import torch

from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import yaml
import tempfile

from datasets import load_dataset

from xenon_generation.models.modeling_gpt2_lab import GPT2LabLMHeadModel, GPT2LabConfig


from peft import PeftModel, PeftConfig


from src.data.feedback_dataset import DataCollatorForFeedbackDataset
from src.tasks.utils import _get_logprob

AutoConfig.register("gpt2lab", GPT2LabConfig)
AutoModelForCausalLM.register(GPT2LabConfig, GPT2LabLMHeadModel)
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    config_path: Optional[str] = field(default="/mnt/bn/st-data-lq/liuzhicheng.lzc/pyneurst/conf/13b_generate.yml")
    num_gpus: Optional[int] = field(default=1)

@dataclass
class DataArguments:
    train_file: str = field(default=None, metadata={"help": "Path to the training data."})
    validation_file: str = field(default=None, metadata={"help": "Path to the validation_file"})

    save_file: str = field(default=None)

    prompt: str = field(default="<srctext> [PLHD95] 请翻译成英文. ")

def get_logprob(input_ids,labels,model,device):
    input_ids = input_ids.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        output = model(input_ids)
    logprobs = _get_logprob(output.logits,labels)
    return  logprobs

@dataclass
class DataCollatorForFeedbackDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        return_dict = {}
        for key in ("A", "A_labels","B", "B_labels"):
            entry = [torch.tensor(instance[key]).long() for instance in instances]
            data = torch.nn.utils.rnn.pad_sequence(
            entry, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            if "labels" in key:
                data[data.eq(self.tokenizer.pad_token_id)] = -100
            return_dict[key] = data

        return return_dict

def make_feedback_data_module(data_args,model_args,tokenizer):
    def preprocess_function(examples):
        prompt = data_args.prompt
        sources = [d.strip() for d in examples["src_text"]]
        A_responses = [d[0].strip() for d in examples["tgt_text"]]
        B_responses = [d[1].strip() for d in examples["tgt_text"]]
        # teacher_responses = [d.strip() for d in examples["better_translation"]]
                
        prefixes = [prompt.replace("<srctext>",src) for src in sources]

        prefix_inputs = tokenizer(prefixes,padding="do_not_pad",truncation=True,max_length=256)
        
        A_inputs = tokenizer(A_responses, max_length=256, padding="do_not_pad",truncation=True)
        B_inputs = tokenizer(B_responses, max_length=256, padding="do_not_pad", truncation=True)
        # teacher_inputs = tokenizer(teacher_responses, max_length=256, padding="do_not_pad", truncation=True)

        A_examples = [
            torch.tensor(s+t + [tokenizer.eos_token_id]).long() for s,t in zip(prefix_inputs['input_ids'],A_inputs['input_ids'])
        ]
        A_examples_labels = [
            torch.tensor([-100] * len(s) + t + [tokenizer.eos_token_id]).long() for s,t in zip(prefix_inputs['input_ids'],A_inputs['input_ids'])
        ]
        B_examples = [
            torch.tensor(s+t + [tokenizer.eos_token_id]).long() for s,t in zip(prefix_inputs['input_ids'],B_inputs['input_ids'])
        ]
        B_examples_labels = [
            torch.tensor([-100] * len(s) + t + [tokenizer.eos_token_id]).long() for s,t in zip(prefix_inputs['input_ids'],B_inputs['input_ids'])
        ]
        # teacher_examples = [
        #     torch.tensor(s+t + [tokenizer.eos_token_id]).long() for s,t in zip(prefix_inputs['input_ids'],teacher_inputs['input_ids'])
        # ]
        # teacher_examples_labels = [
        #     torch.tensor([-100] * len(s) + t + [tokenizer.eos_token_id]).long() for s,t in zip(prefix_inputs['input_ids'],teacher_inputs['input_ids'])
        # ]
        
        return {
            "input_ids": A_examples,
            "A": A_examples,
            "A_labels": A_examples_labels,
            "B": B_examples,
            "B_labels": B_examples_labels,
            # "teacher": teacher_examples,
            # "teacher_labels": teacher_examples_labels,
    }

    data_files = {}
    dataset_args = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
    extension = data_args.train_file.split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)
    column_names = raw_datasets["train"].column_names

    tokenized_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        num_proc=12,
        remove_columns=column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )
    train_dataset = tokenized_datasets["train"]
    data_collator = DataCollatorForFeedbackDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

def get_model(model_args):
    config = yaml.full_load(open(model_args.config_path))
    with tempfile.NamedTemporaryFile(mode="w+") as f:
            f.write(json.dumps(config['model_config'], ensure_ascii=False))
            f.flush()
            config_dict = AutoConfig.from_pretrained(f.name)
            if "adapter_model.bin" in os.listdir(model_args.model_name_or_path):
                print("Loading LoRA Model")
                adapter_config = PeftConfig.from_pretrained(model_args.model_name_or_path)
                model = AutoModelForCausalLM.from_pretrained(adapter_config.base_model_name_or_path,config=config_dict)
                model = PeftModel.from_pretrained(model,model_args.model_name_or_path)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path,config=config_dict)
    model.eval()
    model.half()
    return model

def process_chunk(model_args,data_args,dataset,original_data,data_collator,index,return_dict):
    device = "cuda:{}".format(index)
    model = get_model(model_args).to(device)
    train_dataloader = DataLoader(dataset,batch_size=1,shuffle=False,collate_fn=data_collator)

    correct = 0
    ret = []
    for batch, orig_d in tqdm(zip(train_dataloader,original_data)):
        A_logprob = get_logprob(batch["A"],batch["A_labels"],model,device)
        B_logprob = get_logprob(batch["B"],batch["B_labels"],model,device)

        if orig_d["comparison"] == "A" and A_logprob.item() > B_logprob.item():
            correct += 1
        elif orig_d["comparison"] == "B" and B_logprob.item() > A_logprob.item():
            correct += 1

        ret.append(
            {**orig_d,
            **{
                "A_logprob": A_logprob.item(),
                "B_logprob": B_logprob.item()
            }}
        )
    print(correct / len(original_data))
    return_dict[index] = ret

def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=True)

    model_args.feedback_strategy = "dpo"
    data_module = make_feedback_data_module(data_args,model_args,tokenizer)

    with open(data_args.train_file) as f:
        original_data = json.load(f)
    train_dataset = data_module["train_dataset"]
    data_collator = DataCollatorForFeedbackDataset(tokenizer=tokenizer)

    manager = torch.multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []

    chunk_size = len(train_dataset) // model_args.num_gpus
    for device_num in range(model_args.num_gpus):
        start, end = device_num*chunk_size, min((device_num+1)*chunk_size,len(train_dataset))
        train_subset = torch.utils.data.Subset(train_dataset,
            list(range(start,end))
        )
        p = torch.multiprocessing.Process(target=process_chunk, args=(model_args,data_args,train_subset,original_data[start:end],data_collator, device_num,return_dict))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    final_data = []
    for k in sorted(return_dict):
        data = return_dict[k]
        final_data += data

    with open(data_args.save_file,"w") as fout:
        json.dump(final_data,fout,ensure_ascii=False,indent=4)

if __name__ == "__main__":
    main()

        
    