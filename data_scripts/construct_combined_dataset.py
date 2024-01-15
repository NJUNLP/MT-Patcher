from data_scripts.utils import read_json_or_jsonl, dump_json_or_jsonl
from pipeline.data_manager import SentenceAnalyzerDataManager, CaseGenerationDataManager, FeedbackDataManager, WordAnalogyDataManager
import re


def main(args):
    datas = read_json_or_jsonl(args.infile)
    sa_data = []
    cg_data = []
    wa_data = []

    parse = invalid = 0
    for d in datas:
        sentence_analysis = d["synthesized_case"]
        sa_data.append(
            {
                "prompt": SentenceAnalyzerDataManager.prompt.replace("<src_text>",d["src_text"]),
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
                "prompt": CaseGenerationDataManager.prompt.replace("<domain_topic_style>",sentence_analysis).replace("<word_pair>","{}({})".format(error_source,correction)),
                "response": "Chinese Sentence: {}\nEnglish Sentence: {}".format(src,tgt)
            }
        )

    print("Total {}, parse error {}, invalid {}".format(len(datas),parse,invalid))
    dump_json_or_jsonl(args.sa_outfile, sa_data)
    dump_json_or_jsonl(args.cg_outfile,cg_data)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile")
    parser.add_argument("--sa-outfile")
    parser.add_argument("--cg-outfile")

    args = parser.parse_args()
    main(args)
