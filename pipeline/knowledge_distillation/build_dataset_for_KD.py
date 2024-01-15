from pipeline.data_utils import read_json_or_jsonl
import json
import argparse

def main(args):
    data = read_json_or_jsonl(args.infile)

    parallel_data = []
    for d in data:
        prompt = args.prompt.replace("<src_text>",d["src_text"]).replace("<srclang>",args.srclang).replace("<tgtlang>",args.tgtlang)
        response = d["tgt_text"]
        parallel_data.append(
            {
                "prompt": prompt,
                "response": response
            }
        )
    with open(args.outfile,"w") as fout:
        for d in parallel_data:
            fout.write(json.dumps(d,ensure_ascii=False) + "\n")





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt",default="""Translate the following sentences from <srclang> to <tgtlang>.
Input: <src_text>
Output: """)
    parser.add_argument("--srclang",default="Chinese")
    parser.add_argument("--tgtlang",default="English")
    parser.add_argument("--infile")
    parser.add_argument("--outfile")

    args = parser.parse_args()
    main(args)