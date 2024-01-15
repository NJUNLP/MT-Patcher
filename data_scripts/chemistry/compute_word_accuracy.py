from ast import arg
from data_scripts.utils import read_json_or_jsonl
from collections import defaultdict
import argparse

def compute_test_A(data):
    res = defaultdict(lambda:[0,0])
    for d in data:
        ref = d["tgt_word"].lower()
        res[d["num_context"]][0] += 1
        if ref in d["tgt_text"].lower():
            res[d["num_context"]][1] += 1

    overall_correct = sum(v[1] for v in res.values())
    total = sum(v[0] for v in res.values())

    print("Overall Acc: {:.4f}".format(overall_correct/total))
    for num in res:
        v = res[num]
        print("Acc for {} context: {}".format(num,v[1]/v[0]))

def compute_test_B(data):
    correct = 0
    for d in data:
        ref = d["tgt_word"].lower()
        if ref in d["tgt_text"].lower():
            correct += 1
    print("Acc: {:.4f}".format(correct / len(data)))

def main(args):
    data = read_json_or_jsonl(args.infile)
    if "test-A" in args.infile:
        compute_test_A(data)
    else:
        compute_test_B(data)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile")

    args = parser.parse_args()
    main(args)