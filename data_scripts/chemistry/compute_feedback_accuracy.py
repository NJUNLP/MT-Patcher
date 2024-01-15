from ast import arg
from importlib import invalidate_caches
from data_scripts.utils import read_json_or_jsonl
from collections import defaultdict
import argparse
import re


def main(args):
    data = read_json_or_jsonl(args.infile)
    correct1, correct2, total = 0, 0, len(data)
    invalid = 0
    for d in data:
        if d["model_assessment"].startswith("No Error."):
            if d["tgt_word"].lower() in d["tgt_text"].lower():
                correct1 += 1
        else:
            assessment = d["model_assessment"]
            error_sources = re.findall(r"Error\sSource\sWord:(.+?)\n",assessment)
            corrections = re.findall(r"Correction:(.+?)\n",assessment)
            pe = re.findall(r"Better\sTranslation:(.+)",assessment)
            if len(pe) != 1:
                invalid += 1
                continue
            if len(error_sources) != len(corrections):
                invalid += 1
                continue
            for error_source, correction in zip(error_sources,corrections):
                if error_source.strip() == d["src_word"].strip() and correction.lower().strip() == d["tgt_word"].lower().strip() :
                    correct2 += 1            

    print(correct1,correct2,len(data),invalid)
    print("Feedback Accuracy: {}".format((correct1 + correct2) / total))

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile")

    args = parser.parse_args()
    main(args)