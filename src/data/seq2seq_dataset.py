from functools import partial
from datasets import load_dataset
import torch
from typing import Any,  Dict, NewType,Sequence
InputDataClass = NewType("InputDataClass", Any)
from transformers import DataCollatorForSeq2Seq

from dataclasses import dataclass, field

mapping = {
        "en": "eng_Latn",
        "de": "deu_Latn",
        "fr": "fra_Latn",
        "ca": "cat_Latn",
        "fi": "fin_Latn",
        "ru": "rus_Cyrl",
        "bg": "bul_Cyrl",
        "zh": "zho_Hans",
        "ko": "kor_Hang",
        "ar": "arb_Arab",
        "sw": "swh_Latn",
        "hi": "hin_Deva",
        "ta": "tam_Taml"
    }

def _preprocess_function(tokenizer,examples):
    sources = examples["src_text"]
    targets = examples["tgt_text"]

    result = tokenizer(sources,padding=True,truncation=True,max_length=512)
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=512, padding=True,truncation=True)
    
    result["labels"] = labels["input_ids"]
    return result

def make_seq2seq_data_module(data_args,model_args,tokenizer):
    tokenizer.src_lang = mapping[data_args.srclang]
    tokenizer.tgt_lang = mapping[data_args.tgtlang]

    preprocess_function = partial(_preprocess_function,tokenizer)
    data_files = {}
    dataset_args = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
    extension = data_args.train_file.split(".")[-1]
    if extension == "jsonl":
        extension = "json"
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

    return train_dataset


if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file")
    parser.add_argument("--validation-file")
    parser.add_argument("--srclang", default="zh")
    parser.add_argument("--tgtlang", default="en")
    parser.add_argument("--model-dir")
    parser.add_argument("--preprocessing-num-workers",default=1)
    parser.add_argument("--overwrite-cache",default=True)
    parser.add_argument("--per-device-train-batch-size",default=20,type=int)
    parser.add_argument("--per-device-eval-batch-size",default=20,type=int)


    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    data_module = make_seq2seq_data_module(args,None,tokenizer)

    dataloader = torch.utils.data.DataLoader(data_module["train_dataset"],collate_fn=DataCollatorForSeq2SeqDataset(tokenizer=tokenizer),batch_size=8)

    for batch in dataloader:
        import pdb; pdb.set_trace()