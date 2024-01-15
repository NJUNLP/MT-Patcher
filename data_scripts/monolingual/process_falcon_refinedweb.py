import json
import argparse
import random
from tqdm import tqdm
import pandas as pd

def main(args):

    sents = []
    num_docs = 0
    data = pd.read_parquet(args.infile,engine="pyarrow")
    for d in tqdm(data["content"]):
        num_docs += 1
        sents.extend(
            d.split("\n")
        )

    random.shuffle(sents)
    print("Splitting {} documents to {} sentences".format(num_docs,len(sents)))

    with open(args.outfile,"w") as fout:
        for sent in sents:
            if len(sent.split()) > 128 or len(sent.split()) < 5:
                continue
            fout.write(
                json.dumps(
                    {"src_text": sent},ensure_ascii=False
                ) + "\n"
            )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile")
    parser.add_argument("--outfile")

    args = parser.parse_args()
    main(args)
