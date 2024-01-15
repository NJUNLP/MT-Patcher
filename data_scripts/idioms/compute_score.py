from data_scripts.utils import read_json_or_jsonl
import argparse
import re

def main(args):
    data = read_json_or_jsonl(args.infile)
    scores = []
    invalid = 0
    for d in data:
        try:
            evaluation = d["gpt4_evaluation"]
            score = re.findall(r"Score:\s*(\d)",evaluation)
            scores.append(int(score[0]))
        except:
            invalid += 1
            continue
    print("Scores: {}".format(sum(scores)/len(scores)))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile")

    args = parser.parse_args()
    main(args)