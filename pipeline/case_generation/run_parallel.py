from time import sleep
from src.utils import load_model_and_tokenizer, generate_batch, LMPrefixDataLoader
import re
import json
from tqdm import tqdm
import os
import re
import torch


def read_json(infile,prompt):
    prompts,datas = [],[]
    skipped = 0
    with open(infile) as f:
        for line in f:
            d = json.loads(line)
            src, assessment = d["src_text"].strip(), d["model_assessment"]
            error_words = re.findall("错误词：(.+)\n",assessment)
            correction = re.findall("正确翻译：(.+)",assessment)
            if len(error_words) != 1 or len(correction) != 1:
                skipped += 1
                continue
            else:
                prompt_now = []
                for ew, ec in zip(error_words,correction):
                    prompt_now.append(prompt.replace("<error_span>",ew).replace("<correction>",ec))
                prompts.append(prompt_now[0])
                datas.append(d)
    print("Skipped {}".format(skipped))
    return prompts, datas


def generate_chunks(args,prompts,datas,device_id,return_dict):
    sleep(device_id*30)
    if args.config_dir is not None and args.config_dir.startswith("hdfs"):
        tokenizer_path = config_path = os.path.basename(args.config_dir)
    else:
        tokenizer_path = args.config_dir
        config_path = args.config_dir
    tokenizer, model = load_model_and_tokenizer(args.model_path,tokenizer_path,config_path,device="cuda:{}".format(device_id))
    dataloader = LMPrefixDataLoader(prompts,tokenizer,max_tokens=args.max_tokens)
    
    print("Generating Patch Data")
    cases,ids = [],[]

    for i,batch in enumerate(tqdm(dataloader)):
        completions = generate_batch(model,
            tokenizer,
            batch,
            device="cuda:{}".format(device_id),
            left_pad=True,
            max_new_tokens=256,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            temperature=1.5,
            early_stopping=True)
        ids.extend([b[1] for b in batch])

        parallel_data = []
        for c in completions:
            source_sent = re.findall(r"中文句子:(.+)\n",c)
            target_sent = re.findall(r"英文句子:(.+)\n",c+"\n")
            if len(source_sent) == 2 and len(target_sent) == 2:
                parallel_data.append(
                    {"synthesized_source": source_sent[1].strip(), "synthesized_target": target_sent[1].strip()}
                )
            else:
                parallel_data.append(
                    {
                        "synthesized_source": [],
                        "synthesized_target": []
                    }
                )
        cases.extend(
            parallel_data
        )
    id2results = {}
    for id, result in zip(ids,cases):
        id2results[id] = result

    return_datas = []
    for i,d in enumerate(datas):
        if i in id2results:
            d["generated_cases"] = id2results[i]
            return_datas.append(d)
        else:
            continue

    return_dict[device_id] = return_datas


def main(args):
    
    prompts, datas = read_json(args.infile,args.prompt)

    devices = [int(i) for i in args.devices.split(",")]
    chunk_size = len(prompts) // len(devices)
    torch.multiprocessing.set_start_method("spawn")
    manager = torch.multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    for i, device_id in enumerate(devices):
        start, end = chunk_size*i, min(chunk_size*(i+1),len(prompts))
        sub_prompts, sub_datas = prompts[start:end], datas[start:end]
        p = torch.multiprocessing.Process(target=generate_chunks, args=(args,sub_prompts,sub_datas,device_id,return_dict))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    final_data = []
    for data in return_dict.values():
        final_data += data

    with open(args.savefile,"w") as fout:
        for d in final_data:
            fout.write(json.dumps(d,ensure_ascii=False) + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile")
    parser.add_argument("--savefile")
    parser.add_argument("--prompt",default="""
假设你有着丰富的中文和英文知识，能够生成流畅、自然、多样的平行句对。我会给你一个中文短语P，和它对应的英文翻译Q。请你生成1句中文句子X，以及该句子对应的翻译Y。要求:
1. X中包含中文短语P, Y中包含P对应的英文翻译Q。

示例输入:
P: 到店礼
Q: gift for visiting the store

示例输出:
中文句子: 春节期间，我们商场为每位到店的客户准备了精美的到店礼。
英文句子: During the Spring Festival, we have prepared a beautiful gift for every customer who visits our store.

现在，请处理以下输入:
P: <error_span>
Q: <correction>
""")
    parser.add_argument("--model-path")
    parser.add_argument("--config-dir")
    parser.add_argument("--max-tokens",type=int,default=512)
    parser.add_argument("--devices")
    args = parser.parse_args()
    main(args)



