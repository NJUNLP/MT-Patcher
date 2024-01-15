from bleurt import score
import argparse

def main(args):
    with open(args.ref_file) as f:
        references = [line.strip() for line in f.readlines()]
    with open(args.hyp_file) as f:
        hypothesis = [line.strip() for line in f.readlines()]

    scorer = score.BleurtScorer(args.checkpoint_path)
    scores = scorer.score(references=references, candidates=hypothesis)
    assert isinstance(scores, list) and len(scores) == 1
    print(scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--ref-file")
    parser.add_argument("--hyp-file")

    args = parser.parse_args()
    main(args)