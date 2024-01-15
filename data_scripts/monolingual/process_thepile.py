from sentence_splitter import SentenceSplitter, split_text_into_sentences
import json
import argparse
import random
from tqdm import tqdm

def main(args):
    splitter = SentenceSplitter(language='en')

    sents = []
    num_docs = 0
    with open(args.infile) as f:
        for line in tqdm(f):
            num_docs += 1
            line = line.replace('"{""source"":""the_pile"",""text"":""',"").replace('""}"',"").replace("\\n","\n")
            sents.extend(
                splitter.split(text=line)
            )

    random.shuffle(sents)
    print("Splitting {} documents to {} sentences".format(num_docs,len(sents)))

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
