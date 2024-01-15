from pipeline.data_utils import read_json_or_jsonl
import re
import json
import argparse

def main(args):
    patch_data = set()
    invalid , total = 0,0
    data = read_json_or_jsonl(args.infile)
    for d in data:
        total += 1
        if "pe" in d:
            parallel = (d["src_text"],d["pe"])
            if parallel not in patch_data:
                patch_data.add(parallel)
        if "synthesized_case" in d:
            source_matched = re.findall(r"Sentence:\s*(.+)\n".format(args.srclang),d["synthesized_case"])
            target_matched = re.findall(r"\n.*Sentence:\s*(.+)".format(args.tgtlang),d["synthesized_case"])
            if len(source_matched) != 1 or len(target_matched) != 1:
                invalid += 1
                continue
            parallel = (source_matched[0],target_matched[0])
            if parallel not in patch_data:
                patch_data.add(parallel)

    print(invalid,total)
    with open(args.outfile,"w") as fout:
        for src,tgt in patch_data:
            fout.write(
                json.dumps(
                    {
                        "src_text": src,
                        "tgt_text": tgt
                    },ensure_ascii=False
                ) + "\n"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile")
    parser.add_argument("--outfile")
    parser.add_argument("--srclang")
    parser.add_argument("--tgtlang")

    args = parser.parse_args()
    main(args)