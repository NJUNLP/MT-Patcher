from ltp import StnSplit
import json
import argparse
import random

def main(args):
    with open(args.infile) as f:
        docs = []
        for line in f:
            line = line.replace('"{""source"":""wudao2.0"",""text"":""',"").replace('""}"',"")
            docs.append(line)
    sents = StnSplit().batch_split(docs)

    random.shuffle(sents)
    print("Splitting {} documents to {} sentences".format(len(docs),len(sents)))

    with open(args.outfile,"w") as fout:
        for sent in sents:
            if len(sent) > 256:
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
