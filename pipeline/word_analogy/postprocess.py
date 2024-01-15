import json
import re
from evaluation.data_utils import read_json_or_jsonl, dump_json_or_jsonl
import argparse
from tqdm import tqdm

def parse(d):
    word_analogy = d["word_analogy"]
    category, structure, context = {}, {}, {}
    splited = word_analogy.split("\n")
    try:
        category["description"] = re.findall(r"(.+)相似词",splited[0])[0]
        structure["description"] = re.findall(r"(.+)相似词",splited[1])[0]
        context["description"] = re.findall(r"(.+)相似词",splited[2])[0]
        category["words"] = re.findall(r"相似词[：:]\s*1\.(.+)2\.(.+)3\.(.+)",splited[0])[0]
        structure["words"] = re.findall(r"相似词[：:]\s*1\.(.+)2\.(.+)3\.(.+)",splited[1])[0]
        context["words"] = re.findall(r"相似词[：:]\s*1\.(.+)2\.(.+)3\.(.+)",splited[2])[0]
        if len(category["words"]) != 3 or len(structure["words"]) != 3 or len(context["words"]) != 3:
            print(category["words"])
            print(structure["words"])
            print(context["words"])
            return None
        word_analogy_parsed = {
            "category": category,
            "structure": structure,
            "context": context
        }
        return word_analogy_parsed
    except:
        return None

def main(args):
    datas = read_json_or_jsonl(args.infile)

    results = []
    total = parse_error = 0
    for d in datas:
        total += 1
        word_analogy_parsed = parse(d)
        if word_analogy_parsed is None:
            parse_error += 1
            continue
        else:
            d["word_analogy_parsed"] = word_analogy_parsed
            results.append(d)

    print("Total: {}, Parse Error: {}".format(total,parse_error))
    dump_json_or_jsonl(args.outfile,results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile")
    parser.add_argument("--outfile")

    args = parser.parse_args()
    main(args)

    
