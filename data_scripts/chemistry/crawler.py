import json
import argparse
from copy import deepcopy

def main(args):
    with open(args.infile) as f:
        data = json.load(f)

    ret_d = []
    for d in data:
        if "-" in d["字段2"] or "-" in d["字段1"]:
            continue
        if ".." in d["字段2"]:
            continue
        if "杂质" in d["字段1"]:
            continue
        if "(" in d["字段1"] or "(" in d["字段2"]:
            continue

        new_d = {
            "src_word": d["字段1"],
            "tgt_word": d["字段2"]
        }
        ret_d.append(new_d)
    print(len(ret_d))

    with open(args.outfile,"w") as fout:
        json.dump(ret_d,fout,ensure_ascii=False,indent=2)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile")
    parser.add_argument("--outfile")

    args = parser.parse_args()
    main(args)