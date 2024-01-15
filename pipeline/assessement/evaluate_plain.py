from src.utils import load_model_and_tokenizer, generate
import re
import json
from tqdm import tqdm
import os

def read_data(srcfile,hypfile,prompt):
    prompts = []
    with open(srcfile) as fsrc, open(hypfile) as fhyp:
        for src,tgt in zip(fsrc,fhyp):
            src,tgt = src.strip(), tgt.strip()
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
    prompts = read_data(args.srcfile,args.hypfile,args.prompt)
    
    predicted_scores = []
    responses = []
    print("Generating Assessment")
    for prompt in tqdm(prompts):
        response = generate(model,tokenizer,prompt,do_sample=False,max_new_tokens=512)
        score = postprocess_fn(response)
        predicted_scores.append(score)
        responses.append(response)

    if args.savefile is not None:
        datas = []
        for p,r,s in zip(prompts,responses,predicted_scores):
            datas.append(
                {"prompt": p,
                "response": r.split("\n"),
                "score": s
                }
            )

        with open(args.savefile,"w") as fout:
            json.dump(datas,fout,ensure_ascii=False,indent=4)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--srcfile")
    parser.add_argument("--hypfile")
    parser.add_argument("--savefile")
    parser.add_argument("--prompt",default="You are a translator with professional skills of English and Chinese. Here is a Chinese sentence and its corresponding English translations. Please find possible translation errors in the translation, and the type, severity and explanation of errors. If there is no error, simply answer 'No errors found'. \n Chinese Sentence: <srctext>\n English Sentence: <tgttext>\n Response: ")
    parser.add_argument("--model-path")
    parser.add_argument("--config-dir")

    args = parser.parse_args()
    main(args)



