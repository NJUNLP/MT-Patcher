from data_scripts.utils import read_json_or_jsonl, dump_json_or_jsonl
import argparse

def main(args):
    data = read_json_or_jsonl(args.infile)
    out_data = []

    for i,d in enumerate(data):
        if i%args.N < args.K:
            out_data.append(d)

    dump_json_or_jsonl(args.outfile,out_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile")
    parser.add_argument("--outfile")
    parser.add_argument("--N",type=int)
    parser.add_argument("--K",type=int)

    args = parser.parse_args()
    main(args)
