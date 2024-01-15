from src.utils import load_model_and_tokenizer, generate
import re
import json
from scipy import stats
from tqdm import tqdm

def read_data_from_plain(file_prefix,srclang,tgtlang,prompt):
    prompts = []
    with open(file_prefix+".src") as fsrc, open(file_prefix+".mt") as ftgt, open(file_prefix+".{}-{}.mqm_score.mqm".format(srclang,tgtlang)) as fscore:
        for src,tgt in zip(fsrc,ftgt):
            prompts.append(prompt.replace("<srctext>",src).replace("<tgttext>",tgt))
        scores = [float(s) for s in fscore.readlines()]
    return prompts,scores

def postprocess_fn(text):
    N_major = len(re.findall(r'[mM]ajor',text))
    N_minor = len(re.findall(r'[mM]inor',text))
    return (-1) * N_minor + (-5) * N_major

def main(args):
    tokenizer, model = load_model_and_tokenizer(args.model_path,args.tokenizer_path,args.config_path)
    prompts, scores = read_data_from_plain(args.file_prefix,args.srclang,args.tgtlang,args.prompt)
    
    predicted_scores = []
    responses = []
    print("Generating Assessment")
    for prompt in tqdm(prompts):
        response = generate(model,tokenizer,prompt,do_sample=False,max_new_tokens=512)
        score = postprocess_fn(response)
        predicted_scores.append(score)
        responses.append(response)

    res = stats.spearmanr(scores,predicted_scores)
    print(res)

    if args.savefile is not None:
        datas = []
        import pdb; pdb.set_trace()
        for p,r in zip(prompts,responses):
            datas.append(
                {"prompt": p,
                "response": r.split("\n")}
            )

        with open(args.savefile,"w") as fout:
            json.dump(datas,fout,ensure_ascii=False,indent=4)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-prefix")
    parser.add_argument("--srclang")
    parser.add_argument("--tgtlang")
    parser.add_argument("--savefile")
    parser.add_argument("--prompt",default="You are a translator with professional skills of English and Chinese. Here is a Chinese sentence and its corresponding English translations. Please find possible translation errors in the translation, and the type, severity and explanation of errors. If there is no error, simply answer 'No errors found'. \n Chinese Sentence: <srctext>\n English Sentence: <tgttext>\n Response: ")
    parser.add_argument("--model-path")
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--config-path")

    args = parser.parse_args()
    main(args)



