from functools import partial
from datasets import load_dataset
from accelerate import Accelerator
import torch
from typing import Any,  Dict, NewType,Sequence
InputDataClass = NewType("InputDataClass", Any)
import transformers

from dataclasses import dataclass, field

@dataclass
class DataCollatorForSFTDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        return_dict = {}
        for key in ("input_ids","labels"):
            if key not in instances[0]:
                continue
            entry = [torch.tensor(instance[key]).long() for instance in instances]
            data = torch.nn.utils.rnn.pad_sequence(
                entry, batch_first=True, padding_value=self.tokenizer.pad_token_id,)
            if "labels" in key:
                data[data.eq(self.tokenizer.pad_token_id)] = -100
            return_dict[key] = data
        return_dict["attention_mask"] = return_dict["input_ids"].ne(self.tokenizer.pad_token_id)
        return return_dict

def _preprocess_function(tokenizer,examples):
    str_prompts = examples["prompt"] if "prompt" in examples else examples["src_text"]
    str_responses = examples["response"] if "response" in examples else examples["tgt_text"]

    tokenized_prompts = tokenizer(str_prompts,padding="do_not_pad",truncation=True,max_length=512)
    
    tokenized_responses = tokenizer(str_responses, max_length=512, padding="do_not_pad",truncation=True)

    input_ids = [
        torch.tensor(s+t + [tokenizer.eos_token_id]).long() for s,t in zip(tokenized_prompts['input_ids'],tokenized_responses['input_ids'])
    ]
    labels = [
        torch.tensor([-100] * len(s) + t + [tokenizer.eos_token_id]).long() for s,t in zip(tokenized_prompts['input_ids'],tokenized_responses['input_ids'])
    ]
    
    return {
        "input_ids": input_ids,
        "labels": labels
    }

def make_sft_data_module(data_args,model_args,tokenizer):
    preprocess_function = partial(_preprocess_function,tokenizer)
    data_files = {}
    dataset_args = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
    raw_datasets = load_dataset("json", data_files=data_files, **dataset_args)
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
    data_collator = DataCollatorForSFTDataset(tokenizer=tokenizer)

    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file")
    parser.add_argument("--validation-file")
    parser.add_argument("--prompt",default="<srctext> [PLHD95] 请翻译成英文. ")
    parser.add_argument("--srclang", default="Chinese")
    parser.add_argument("--tgtlang", default="English")
    parser.add_argument("--model-dir")
    parser.add_argument("--preprocessing-num-workers",default=1)
    parser.add_argument("--overwrite-cache",default=True)
    parser.add_argument("--per-device-train-batch-size",default=20,type=int)
    parser.add_argument("--per-device-eval-batch-size",default=20,type=int)


    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    accelerator = Accelerator()
    data_module = make_feedback_data_module(args,tokenizer,accelerator)

    for batch in data_module["train_dataloader"]:
        import pdb; pdb.set_trace()