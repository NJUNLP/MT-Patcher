from data_scripts.utils import read_json_or_jsonl, dump_json_or_jsonl
from pipeline.data_manager import WordAnalogyDataManager
import re


def main(args):
    datas = read_json_or_jsonl(args.infile)
    wa_data = []
    parse = invalid = 0
    for d in datas:
        try:
            error_source = d["error_sources"][0]
            if len(error_source.split()) > 10:
                invalid += 1
                continue
            src = d["src_text"]
            word_analogy = d["word_analogy"]

            wa_data.append(
            {
                "prompt": WordAnalogyDataManager.prompt.replace("<src_text>",src).replace("<error_word>",error_source).replace("<srclang>",args.srclang).replace("<tgtlang>",args.tgtlang),
                "response": word_analogy
            }
        )
        except:
            invalid += 1
            continue
        

    print("Total {}, parse error {}, invalid {}".format(len(datas),parse,invalid))
    dump_json_or_jsonl(args.savedir+"/wa.jsonl", wa_data)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile")
    parser.add_argument("--srclang")
    parser.add_argument("--tgtlang")
    parser.add_argument("--savedir")

    args = parser.parse_args()
    main(args)
