from src.utils import load_model_and_tokenizer, generate
import re
import json
from tqdm import tqdm
import os
import re

def read_data(infile,prompt):
    prompts = []
    with open(infile) as f:
        for line in f:
            d = json.loads(line)
            src,tgt = d["src_text"].strip(), d["tgt_text"][0].strip()
            prompts.append(prompt.replace("<srctext>",src).replace("<tgttext>",tgt))
    return prompts

def postprocess_fn(text):
    N_major = len(re.findall(r'[mM]ajor',text))
    N_minor = len(re.findall(r'[mM]inor',text))
    return (-1) * N_minor + (-5) * N_major


def main(args):
    if args.config_dir is not None and args.config_dir.startswith("hdfs"):
        tokenizer_path = config_path = os.path.basename(args.config_dir)
    else:
        tokenizer_path = args.config_dir
        config_path = args.config_dir
    tokenizer, model = load_model_and_tokenizer(args.model_path,tokenizer_path,config_path)
    prompts = read_data(args.infile,args.prompt)
    
    assessments = []
    print("Generating Assessment")
    for prompt in tqdm(prompts):
        response = generate(model,tokenizer,prompt,do_sample=False,max_new_tokens=512)
        splited = re.split("评估:\n",response,1)
        assessment = splited[1]
        assessments.append(assessment)

    if args.savefile is not None:
        datas = []
        with open(args.infile) as f:
            for line,r in zip(f,assessments):
                d = json.loads(line)
                d["model_assessment"] = r
                datas.append(d)

        with open(args.savefile,"w") as fout:
            json.dump(datas,fout,ensure_ascii=False,indent=4)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile")
    parser.add_argument("--savefile")
    parser.add_argument("--prompt",default="""
假设你是一个非常专业的翻译人员，擅长对机器翻译的结果给出详细完整的评估。 我会给你一句中文句子X和它的英文翻译Y，请你帮忙评估该翻译。
1. 你应该首先给出总体评价。
2. 紧接着，如果有错误，请给出错误，并加以解释。如果没有错误，就不用给出解释。
3. 在解释错误的时候，要求给出错误对应原文片段，错误在译文中的位置，以及错误的理由。
4. 对于多个错误，你应该分条说明。尽量抽取出包含错误的最小片段，并加以说明，避免出现错误位置是整个句子的情况。
5. 你的回答应该是中文。

中文原文: <srctext>
英文译文: <tgttext>
评估:
""")
    parser.add_argument("--model-path")
    parser.add_argument("--config-dir")
    args = parser.parse_args()
    main(args)



