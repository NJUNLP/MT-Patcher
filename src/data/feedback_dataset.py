from functools import partial
from transformers import default_data_collator
from datasets import load_dataset
from torch.utils.data import DataLoader
from accelerate import Accelerator
import torch
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union,Sequence
InputDataClass = NewType("InputDataClass", Any)
from collections.abc import Mapping
import numpy as np
import transformers

from dataclasses import dataclass, field

@dataclass
class DataCollatorForFeedbackDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        return_dict = {}
        for key in ("good_examples", "good_examples_labels","bad_examples", "bad_examples_labels", "teacher_examples", "teacher_examples_labels","ref_good","ref_bad"):
            if key not in instances[0]:
                continue
            if key not in ["ref_good","ref_bad"]:
                entry = [torch.tensor(instance[key]).long() for instance in instances]
                data = torch.nn.utils.rnn.pad_sequence(
                entry, batch_first=True, padding_value=self.tokenizer.pad_token_id)
                if "labels" in key:
                    data[data.eq(self.tokenizer.pad_token_id)] = -100
            else:
                data = torch.tensor([instance[key] for instance in instances]).float()
            return_dict[key] = data

        return return_dict

def preprocess_BC(data_args,tokenizer,examples):
    prompt = data_args.prompt
    sources = examples["src_text"]
    _responses = examples["tgt_text"]
    _comparisons = examples["comparison"]
    teacher_responses = examples["better_translation"]        

    oracle_responses = []

    if data_args.BC_oracle == "teacher":
        oracle_responses = teacher_responses
    else:
        for response_list, comp in zip(_responses, _comparisons):
            if comp == "A":
                oracle_responses.append(response_list[0])
            elif comp == "B":
                oracle_responses.append(response_list[1])
            else:
                continue
            
    prefixes = [prompt.replace("<srctext>",src) for src in sources]

    prefix_inputs = tokenizer(prefixes,padding="do_not_pad",truncation=True,max_length=256)
    
    oracle_inputs = tokenizer(oracle_responses, max_length=256, padding="do_not_pad",truncation=True)

    oracle_examples = [
        torch.tensor(s+t + [tokenizer.eos_token_id]).long() for s,t in zip(prefix_inputs['input_ids'],oracle_inputs['input_ids'])
    ]
    oracle_examples_labels = [
        torch.tensor([-100] * len(s) + t + [tokenizer.eos_token_id]).long() for s,t in zip(prefix_inputs['input_ids'],oracle_inputs['input_ids'])
    ]
    
    return {
        "input_ids": oracle_examples_labels,
        "good_examples": oracle_examples,
        "good_examples_labels": oracle_examples_labels,
}

def preprocess_CBC(data_args,tokenizer,examples):
    prompt = data_args.prompt
    conditional_good = "[高质量翻译]: "
    conditional_bad = "[低质量翻译]: "
    sources = examples["src_text"]
    _responses = examples["tgt_text"]
    _comparisons = examples["comparison"]
    teacher_responses = examples["better_translation"]

    good_responses, bad_responses = [], []
    for response_list, comp in zip(_responses, _comparisons):
        if comp == "A":
            good_responses.append(conditional_good + response_list[0])
            bad_responses.append(conditional_bad + response_list[1])
        elif comp == "B":
            good_responses.append(conditional_good + response_list[1])
            bad_responses.append(conditional_bad + response_list[0])

    if data_args.CBC_add_teacher_comparison:
        for response_list, teacher_response, comp in zip(_responses,teacher_responses,_comparisons):
            if comp == "C":
                good_responses.append(conditional_good + teacher_response)
                bad_responses.append(conditional_bad + response_list[0])

                good_responses.append(conditional_good + teacher_response)
                bad_responses.append(conditional_bad + response_list[1])
            
    prefixes = [prompt.replace("<srctext>",src) for src in sources]

    prefix_inputs = tokenizer(prefixes,padding="do_not_pad",truncation=True,max_length=256)
    
    good_inputs = tokenizer(good_responses, max_length=256, padding="do_not_pad",truncation=True)
    bad_inputs = tokenizer(bad_responses, max_length=256, padding="do_not_pad", truncation=True)

    good_examples = [
        torch.tensor(s+t + [tokenizer.eos_token_id]).long() for s,t in zip(prefix_inputs['input_ids'],good_inputs['input_ids'])
    ]
    good_examples_labels = [
        torch.tensor([-100] * len(s) + t + [tokenizer.eos_token_id]).long() for s,t in zip(prefix_inputs['input_ids'],good_inputs['input_ids'])
    ]
    bad_examples = [
        torch.tensor(s+t + [tokenizer.eos_token_id]).long() for s,t in zip(prefix_inputs['input_ids'],bad_inputs['input_ids'])
    ]
    bad_examples_labels = [
        torch.tensor([-100] * len(s) + t + [tokenizer.eos_token_id]).long() for s,t in zip(prefix_inputs['input_ids'],bad_inputs['input_ids'])
    ]
    
    return {
        "input_ids": good_examples_labels,
        "good_examples": good_examples,
        "good_examples_labels": good_examples_labels,
        "bad_examples": bad_examples,
        "bad_examples_labels": bad_examples_labels,
}

def preprocess_DPO(data_args,tokenizer,examples):
    prompt = data_args.prompt
    sources = examples["src_text"]
    _responses = examples["tgt_text"]
    _comparisons = examples["comparison"]
    _A_logprobs = examples["A_logprob"]
    _B_logprobs = examples["B_logprob"]

    good_responses, bad_responses, good_logprobs, bad_logprobs = [], [], [], []
    for response_list, comp, A_logp, B_logp in zip(_responses,  _comparisons, _A_logprobs, _B_logprobs):
        if comp == "A":
            good_responses.append(response_list[0])
            bad_responses.append(response_list[1])
            good_logprobs.append(A_logp)
            bad_logprobs.append(B_logp)
        elif comp == "B":
            good_responses.append(response_list[1])
            bad_responses.append(response_list[0])
            good_logprobs.append(B_logp)
            bad_logprobs.append(A_logp)
        else:
            continue

    if data_args.CBC_add_teacher_comparison:
        teacher_responses = examples["better_translation"]
        _teacher_logprobs = examples["teacher_logprob"]

        for response_list, teacher_response, comp, A_logp, B_logp, teacher_logp in zip(_responses,teacher_responses,_comparisons, _A_logprobs, _B_logprobs, _teacher_logprobs):
            if comp == "C":
                good_responses.append(teacher_response)
                bad_responses.append(response_list[0])
                good_logprobs.append(teacher_logp)
                bad_logprobs.append(A_logp)

                good_responses.append(teacher_response)
                bad_responses.append(response_list[1])
                good_logprobs.append(teacher_logp)
                bad_logprobs.append(B_logp)

    prefixes = [prompt.replace("<srctext>",src) for src in sources]

    prefix_inputs = tokenizer(prefixes,padding="do_not_pad",truncation=True,max_length=256)
    
    good_inputs = tokenizer(good_responses, max_length=256, padding="do_not_pad",truncation=True)
    bad_inputs = tokenizer(bad_responses, max_length=256, padding="do_not_pad", truncation=True)

    good_examples = [
        torch.tensor(s+t + [tokenizer.eos_token_id]).long() for s,t in zip(prefix_inputs['input_ids'],good_inputs['input_ids'])
    ]
    good_examples_labels = [
        torch.tensor([-100] * len(s) + t + [tokenizer.eos_token_id]).long() for s,t in zip(prefix_inputs['input_ids'],good_inputs['input_ids'])
    ]
    bad_examples = [
        torch.tensor(s+t + [tokenizer.eos_token_id]).long() for s,t in zip(prefix_inputs['input_ids'],bad_inputs['input_ids'])
    ]
    bad_examples_labels = [
        torch.tensor([-100] * len(s) + t + [tokenizer.eos_token_id]).long() for s,t in zip(prefix_inputs['input_ids'],bad_inputs['input_ids'])
    ]
    
    return {
        "input_ids": good_examples_labels,
        "good_examples": good_examples,
        "good_examples_labels": good_examples_labels,
        "bad_examples": bad_examples,
        "bad_examples_labels": bad_examples_labels,
        "ref_good": good_logprobs,
        "ref_bad": bad_logprobs
}

def make_feedback_data_module(data_args,model_args,tokenizer):

    if data_args.feedback_strategy == "BC":
        preprocess_function = partial(preprocess_BC,data_args,tokenizer)
    elif data_args.feedback_strategy == "CBC":
        preprocess_function = partial(preprocess_CBC,data_args,tokenizer)
    elif data_args.feedback_strategy == "DPO":
        preprocess_function = partial(preprocess_DPO,data_args,tokenizer)
    else:
        raise ValueError("Unknown Feedback Strategy")
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