from ast import dump
from data_scripts.utils import read_json_or_jsonl, dump_json_or_jsonl
import argparse
import re

def main(args):
    data = read_json_or_jsonl(args.infile)
    ret_data = []
    invalid = 0
    for d in data:
        # try:
            src_text = re.findall(r"Chinese\sSentence:\s(.+)\n",d["synthesized_case"])[0]
            ret_data.append(
                {
                    "src_text": src_text,
                    "definition": d["definition"],
                    "src_word": d["src_word"]
                }
            )

    print(len(ret_data))
    dump_json_or_jsonl(args.outfile,ret_data)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile")
    parser.add_argument("--outfile")

    args = parser.parse_args()
    main(args)