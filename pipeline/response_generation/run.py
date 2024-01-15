from src.utils import load_model_and_tokenizer, generate_batch, LMPrefixDataLoader
import re
import json
from tqdm import tqdm
import os
import re
import torch


def read_data(infile,prompt):
    prompts,datas = [],[]
    with open(infile) as f:
        for line in f:
            line = line.encode("utf-8")
            d = json.loads(line)
            src= d["src_text"].strip()
            prompts.append(prompt.replace("<srctext>",src))
            datas.append(d)
    return prompts, datas    


def main(args):
    if args.config_dir is not None and args.config_dir.startswith("hdfs"):
        tokenizer_path = config_path = os.path.basename(args.config_dir)
    else:
        tokenizer_path = args.config_dir
        config_path = args.config_dir
    tokenizer, model = load_model_and_tokenizer(args.model_path,tokenizer_path,config_path)
    prompts, datas = read_data(args.infile,args.prompt)
    dataloader = LMPrefixDataLoader(prompts,tokenizer,max_tokens=args.max_tokens)
    
    print("Generating Translation")
    translations,ids = [],[]

    for i,batch in enumerate(tqdm(dataloader)):
        completions = generate_batch(model,
            tokenizer,
            batch,
            left_pad=True,
            max_new_tokens=256,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
            early_stopping=True)
        ids.extend([b[1] for b in batch])

        translations.extend([re.findall(r"请翻译成英文：(.+)",c)[0].replace("\n","[PLHD50]").replace("[EOS]","").split("[PLHD50]")[0].strip() for c in completions])
    id2translation = {}
    for id, translation in zip(ids,translations):
        id2translation[id] = translation

    return_datas = []
    for i,d in enumerate(datas):
        if i in id2translation:
            d["tgt_text"] = [id2translation[i]]
            return_datas.append(d)
        else:
            continue

    if args.savefile is not None:
        with open(args.savefile,"w",encoding="utf-8") as fout:
            for d in return_datas:
                fout.write(json.dumps(d,ensure_ascii=False) + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile")
    parser.add_argument("--savefile")
    parser.add_argument("--prompt",default="<srctext> [PLHD95] 请翻译成英文： ")
    parser.add_argument("--model-path")
    parser.add_argument("--config-dir")
    parser.add_argument("--max-tokens",type=int,default=512)
    args = parser.parse_args()
    main(args)



