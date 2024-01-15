import json
from data_scripts.utils import read_json_or_jsonl
import re
import argparse
import random
import copy


def main(args):
    # random.seed(1)
    data = read_json_or_jsonl(args.infile)

    parallel_pairs = []
    pattern = r"\d\.\s*Chinese: .*?\n\s*English: .*?\n"
    for d in data:
        synthesized = d["synthesized_case"]
        pairs = re.findall(pattern,synthesized)
        cur_pairs = []
        for pair in pairs:
            chinese = re.findall(r"Chinese:\s*(.*)\n",pair)[0]
            english = re.findall(r"English:\s*(.*)\n",pair)[0]
            cur_pairs.append(({
                "src_word": d["src_word"],
                "tgt_word": d["tgt_word"],
                "src_text": chinese,
                "ref_text": english
            }))
        parallel_pairs.append(cur_pairs)

    # first split word by train/valid
    random.shuffle(parallel_pairs)
    train_candidate_set, word_generalization_set = parallel_pairs[:-args.num_test], parallel_pairs[-args.num_test:]
    word_generalization_set = [x for xs in word_generalization_set for x in xs]
    # then create context generalization set
    context_generalization_set = []
    train_set = []
    for pairs in train_candidate_set:
        random.shuffle(pairs)
        num = random.randint(1,len(pairs) - 1)
        pairs[-1]["num_context"] = num
        context_generalization_set.append(pairs[-1])
        for p in pairs[:num]:
            _p = copy.deepcopy(p)
            _p["num_context"] = num
            train_set.append(_p)

    print(len(train_set))
    print(len(word_generalization_set))
    print(len(context_generalization_set))

    with open(args.savedir+"/test-A.jsonl","w") as fout:
        for d in context_generalization_set:
            fout.write(json.dumps(d,ensure_ascii=False) + "\n")

    with open(args.savedir + "/test-B.jsonl","w") as fout:
        for d in word_generalization_set:
            fout.write(json.dumps(d,ensure_ascii=False) + "\n")

    with open(args.savedir + "/train.jsonl","w") as fout:
        for d in train_set:
            fout.write(json.dumps(d,ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile")
    parser.add_argument("--num-test",type=int)
    parser.add_argument("--savedir")

    args = parser.parse_args()
    main(args)
