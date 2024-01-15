from time import sleep
from src.utils import load_model_and_tokenizer, LMPrefixDataLoader, collate_tokens
from pipeline.data_utils import dump_json_or_jsonl
import re
import json
from tqdm import tqdm
import os
import re
import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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
        "ta": "tam_Taml",
        "jp": "jpn_Jpan"
    }

def generate_batch(model,tokenizer,batch,device="cuda",**kwargs):
    input = collate_tokens([b[0] for b in batch],pad_idx=tokenizer.pad_token_id).to(device)
    attention_mask = input.ne(tokenizer.pad_token_id)

    encoding = {'input_ids':input, "attention_mask": attention_mask}
    
    completions = []
    with torch.no_grad():
        out_ids = model.generate(
            **encoding,
            **kwargs
        )
        completions = tokenizer.batch_decode(out_ids,skip_special_tokens=True)
    return completions

def generate_chunks(args,prompts,datas,device_id,return_dict):
    sleep(device_id)
    device = "cuda:{}".format(device_id)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path,low_cpu_mem_usage=True,).half().to(device)
    model.eval()

    tokenizer.src_lang = mapping[args.srclang]
    tokenizer.tgt_lang = mapping[args.tgtlang]

    dataloader = LMPrefixDataLoader(prompts,tokenizer,max_tokens=args.max_tokens)
    print("Generating Translations")
    results,ids = [],[]

    for i,batch in enumerate(tqdm(dataloader)):
        completions = generate_batch(model,
            tokenizer,
            batch,
            forced_bos_token_id=tokenizer.lang_code_to_id[mapping[args.tgtlang]],
            device="cuda:{}".format(device_id),
            max_new_tokens=256,
            num_beams=4,
            early_stopping=True)
        ids.extend([b[1] for b in batch])
        results.extend(
            [c for c in completions]
        )

    results = [e for _,e in sorted(zip(ids,results),key=lambda pair: pair[0])]

    return_datas = []
    output_key = args.output_key
    for r,d in zip(results,datas):
        d[output_key] = r
        return_datas.append(d)

    return_dict[device_id] = return_datas

def load_sources(infile):
    sources, datas = [], []
    with open(infile) as f:
        for line in f:
            d = json.loads(line)
            sources.append(d["src_text"])
            datas.append(d)
    return sources, datas

def main(args):
    process_fn =  generate_chunks    
    prompts, datas = load_sources(args.infile)

    devices = [int(i) for i in args.devices.split(",")]
    if len(devices) == 1:
        return_dict = {}
        process_fn(args,prompts,datas,0,return_dict)
    else:
        chunk_size = len(prompts) // len(devices)
        torch.multiprocessing.set_start_method("spawn")
        manager = torch.multiprocessing.Manager()
        return_dict = manager.dict()
        jobs = []
        for i, device_id in enumerate(devices):
            start, end = chunk_size*i, min(chunk_size*(i+1),len(prompts))
            sub_prompts, sub_datas = prompts[start:end], datas[start:end]
            p = torch.multiprocessing.Process(target=process_fn, args=(args,sub_prompts,sub_datas,device_id,return_dict))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()

    final_data = []
    for data in return_dict.values():
        final_data += data

    dump_json_or_jsonl(args.savefile,final_data)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile")
    parser.add_argument("--srclang")
    parser.add_argument("--tgtlang")
    parser.add_argument("--savefile")
    parser.add_argument("--model-path")
    parser.add_argument("--max-tokens",type=int,default=512)
    parser.add_argument("--devices")
    parser.add_argument("--output-key",default="tgt_text")
    args = parser.parse_args()
    main(args)



