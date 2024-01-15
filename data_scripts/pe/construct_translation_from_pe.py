import json
import argparse
from transformers import AutoTokenizer
import random
from tqdm import tqdm
from data_scripts.utils import read_json_or_jsonl

def is_over_length(tokenizer,sentence, max_length):
    input_ids = tokenizer(sentence)["input_ids"]
    if len(input_ids) > max_length:
        return True
    else:
        return False

def main(args):
    data = read_json_or_jsonl(args.infile)

    skipped = 0
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path,trust_remote_code=True)
    parallel_data = []
    for d in data:
        if "pe" not in d:
            src, tgt = d["src_text"], d["tgt_text"]
        else:
            src,tgt = d["src_text"], d["pe"]
        prompt = args.prompt.replace("<src_text>",src.strip()).replace("<srclang>",args.srclang).replace("<tgtlang>",args.tgtlang)
        response = tgt.strip()
        if is_over_length(tokenizer,prompt,args.max_length) or is_over_length(tokenizer,response,args.max_length):
            skipped += 1
            continue
        parallel_data.append(
            {"prompt": prompt,
            "response": tgt.strip()}
        )
    print("Skipped: {}/{}".format(skipped,len(data)))
    with open(args.outfile,"w") as fout:
        for d in parallel_data:
            fout.write(json.dumps(d,ensure_ascii=False) + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile")
    parser.add_argument("--outfile")
    parser.add_argument("--srclang",default="Chinese")
    parser.add_argument("--tgtlang",default="English")
    parser.add_argument("--prompt",default="""Translate the following sentences from <srclang> to <tgtlang>.
Input: <src_text>
Output: 
""")
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--max-length",type=int)

    args = parser.parse_args()
    main(args)
