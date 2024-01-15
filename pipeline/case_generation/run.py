from src.utils import load_model_and_tokenizer, generate_batch, LMPrefixDataLoader
import re
import json
from tqdm import tqdm
import os
import re

def read_json(infile,prompt):
    prompts,datas = [],[]
    with open(infile) as f:
        for line in f:
            d = json.loads(line)
            src, assessment = d["src_text"].strip(), d["model_assessment"]
            error_words = re.findall("错误词：(.+)\n",assessment)
            correction = re.findall("正确翻译：(.+)",assessment)
            if len(error_words) == 0:
                continue
            else:
                prompt_now = []
                for ew, ec in zip(error_words,correction):
                    prompt_now.append(prompt.replace("<error_span>",ew).replace("<correction>",ec))
                prompts.append(prompt_now[0])
                datas.append(d)
    return prompts, datas


def main(args):

    prompts, datas = read_json(args.infile,args.prompt)

    if args.config_dir is not None and args.config_dir.startswith("hdfs"):
        tokenizer_path = config_path = os.path.basename(args.config_dir)
    else:
        tokenizer_path = args.config_dir
        config_path = args.config_dir
    tokenizer, model = load_model_and_tokenizer(args.model_path,tokenizer_path,config_path)
    
    dataloader = LMPrefixDataLoader(prompts,tokenizer,max_tokens=args.max_tokens)
    
    print("Generating Translation")
    cases,ids = [],[]
    for batch in tqdm(dataloader):
        completions = generate_batch(model,
            tokenizer,
            batch,
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
    cases = [e for _,e in sorted(zip(ids,cases),key=lambda pair: pair[0])]

    if args.savefile is not None:
        save_datas = []
        for d,c in zip(datas,cases):
            d["generated_cases"] = c
            save_datas.append(d)
        
        with open(args.savefile,"w",encoding="utf-8") as fout:
            for d in save_datas:
                fout.write(json.dumps(d,ensure_ascii=False) + "\n")

        with open(args.savefile.replace("jsonl","json"),"w",encoding="utf-8") as fout:
            json.dump(save_datas,fout,indent=4,ensure_ascii=False)



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
    args = parser.parse_args()
    main(args)



