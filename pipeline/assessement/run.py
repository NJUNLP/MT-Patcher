from src.utils import load_model_and_tokenizer, generate_batch, LMPrefixDataLoader
import re
import json
from tqdm import tqdm
import os
import re
# from vllm import SamplingParams

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
    prompts, datas = read_data(args.infile,args.prompt)

    # sampling_params = SamplingParams(temperature=0, top_p=1)
    # for idx in range(0,len(prompts),16):
    #     batch = prompts[idx:min(idx+16,len(prompts))]
    #     outputs = model.generate(batch,sampling_params)
    #     for output in outputs:
    #         prompt = output.prompt
    #         generated_text = output.outputs[0].text
    #         print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    
    dataloader = LMPrefixDataLoader(prompts,tokenizer,max_tokens=args.max_tokens)
    
    print("Generating Assessment")
    assessments,ids = [],[]
    for batch in tqdm(dataloader):
        completions = generate_batch(model,
            tokenizer,
            batch,
            left_pad=True,
            max_new_tokens=512,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
            early_stopping=True)
        ids.extend([b[1] for b in batch])
        assessments.extend(
            [c.split("评估:",1)[1].strip() for c in completions]
        )

    id2assessment = {}
    for id, assessment in zip(ids,assessments):
        id2assessment[id] = assessment

    # sorted_assessments = []
    # for i in range(len(prompts)):
    # assessments = [e for _,e in sorted(zip(ids,assessments),key=lambda pair: pair[0])]

    if args.savefile is not None:
        save_datas = []
        for i,d in enumerate(datas):
            if i in id2assessment:
                d["model_assessment"] = id2assessment[i]
                save_datas.append(d)
            else:
                continue
        # for d,r in zip(datas,assessments):
        #     d["model_assessment"] = r
        #     save_datas.append(d)

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
    args = parser.parse_args()
    main(args)



