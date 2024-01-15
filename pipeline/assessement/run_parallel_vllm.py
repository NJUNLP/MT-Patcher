from time import sleep
from src.utils import load_model_and_tokenizer, generate_batch, LMPrefixDataLoader
import re
import json
from tqdm import tqdm
import os
import re
import torch


def read_data(infile,prompt):
    prompts = []
    datas = []
    with open(infile) as f:
        for line in f:
            line = line.encode("utf-8")
            d = json.loads(line)
            src,tgt = d["src_text"].strip(), d["tgt_text"][0].strip()
            prompts.append(prompt.replace("<srctext>",src).replace("<tgttext>",tgt))
            datas.append(d)
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
    
    print("Generating Assessment")
    results,ids = [],[]

    for i,batch in enumerate(tqdm(dataloader)):
        completions = generate_batch(model,
            tokenizer,
            batch,
            device="cuda:{}".format(device_id),
            left_pad=True,
            max_new_tokens=256,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
            early_stopping=True)
        ids.extend([b[1] for b in batch])

        results.extend(
            [c.split("评估:",1)[1].strip() for c in completions]
            )
    id2results = {}
    for id, result in zip(ids,results):
        id2results[id] = result

    return_datas = []
    for i,d in enumerate(datas):
        if i in id2results:
            d["model_assessment"] = id2results[i]
            return_datas.append(d)
        else:
            continue

    return_dict[device_id] = return_datas


def main(args):
    
    prompts, datas = read_data(args.infile,args.prompt)

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
假设你是一个非常专业的翻译人员，擅长对机器翻译的结果给出详细完整的评估。 我会给你一句中文句子X和它的英文翻译Y，请你帮忙评估该翻译。
1. 你应该首先给出总体评价。
2. 紧接着，如果有错误，请给出错误，并加以解释。如果没有错误，就不用给出解释。
3. 在解释错误的时候，要求给出错误对应原文片段，错误在译文中的位置，错误的理由, 错误词以及正确的翻译。
4. 对于多个错误，你应该分条说明。尽量抽取出包含错误的最小片段，并加以说明，避免出现错误位置是整个句子的情况。
5. 你的回答应该是中文。

中文原文: <srctext>
英文译文: <tgttext>
评估:
""")
    parser.add_argument("--model-path")
    parser.add_argument("--config-dir")
    parser.add_argument("--max-tokens",type=int,default=512)
    parser.add_argument("--devices")
    args = parser.parse_args()
    main(args)



