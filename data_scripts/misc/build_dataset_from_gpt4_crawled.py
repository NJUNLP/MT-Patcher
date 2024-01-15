from data_scripts.utils import read_json_or_jsonl, dump_json_or_jsonl
from pipeline.data_manager import SentenceAnalyzerDataManager, CaseGenerationDataManager, WordAnalogyDataManager
import re


def main(args):
    datas = read_json_or_jsonl(args.infile)
    sa_data = []
    cg_data = []

    parse = invalid = 0
    for d in datas:
        sentence_analysis = d["synthesized_case"]
        sa_data.append(
            {
                "prompt": SentenceAnalyzerDataManager.prompt.replace("<src_text>",d["src_text"]).replace("<srclang>",args.srclang).replace("<tgtlang>",args.tgtlang),
                "response": sentence_analysis
            }
        )
        error_source = d["error_sources"][0]
        correction = d["corrections"][0]

        src = d["src_text"]
        tgt = d["pe"]

        if error_source not in src or correction not in tgt:
            invalid += 1
            continue
        cg_data.append(
            {
                "prompt": CaseGenerationDataManager.prompt.replace("<domain_topic_style>",sentence_analysis).replace("<word_pair>","{}({})".format(error_source,correction)).replace("<srclang>",args.srclang).replace("<tgtlang>",args.tgtlang),
                "response": "<srclang> Sentence: {}\n<tgtlang> Sentence: {}".format(src,tgt).replace("<srclang>",args.srclang).replace("<tgtlang>",args.tgtlang)
            }
        )

    print("Total {}, parse error {}, invalid {}".format(len(datas),parse,invalid))
    dump_json_or_jsonl(args.savedir+"/sa.jsonl", sa_data)
    dump_json_or_jsonl(args.savedir+"/cg.jsonl",cg_data)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile")
    parser.add_argument("--srclang")
    parser.add_argument("--tgtlang")
    parser.add_argument("--savedir")

    args = parser.parse_args()
    main(args)
