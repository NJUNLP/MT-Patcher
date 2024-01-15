import json
import random
from pipeline.data_manager import FeedbackDataManager

def read_file(filename):
    with open(filename) as f:
        datas = json.load(f)
    for d in datas:
        if "trg_text" in d:
            d["tgt_text"] = [d["trg_text"]]
    print(filename, len(datas))
    return datas


def main(args):
    ret = []
    positive, negative = 0,0
    for filename in args.infiles:
        datas = read_file(filename)
        for d in datas:
            prompt = FeedbackDataManager.prompt.replace("<srctext>",d["src_text"]).replace("<tgttext>",d["tgt_text"][0] if isinstance(d["tgt_text"],list) else d["tgt_text"]).replace("<srclang>",args.srclang).replace("<tgtlang>",args.tgtlang)
            if len(d["reasons"]) == 0:
                response = "No Error."
                positive += 1
            else:
                response = []
                for i,(location, reason, correction) in enumerate(zip(d["error_sources"],d["reasons"],d["corrections"])):
                    response.append(
                        args.response_template.replace("<num>",str(i+1)).replace("<location>",location).replace("<reason>",reason).replace("<correction>",correction)
                    )
                response = "There are errors in the translation。\n" + "\n".join(response) + "\nBetter Translation: {}".format(d["pe"])
                negative += 1
            ret.append(
                {
                    "prompt": prompt,
                    "response": response
                }
            )

    print("Positive: {}, Negative: {}".format(positive, negative))
    random.shuffle(ret)
    with open(args.outfile,"w") as fout:
        for d in ret:
            fout.write(json.dumps(d,ensure_ascii=False,) + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--infiles",nargs="+")
    parser.add_argument("--outfile")
    parser.add_argument("--srclang")
    parser.add_argument("--tgtlang")
    parser.add_argument("--response-template",default="""Error<num>
Error Source Word: <location>
Explanation: <reason>
Correction: <correction>
""")

    args = parser.parse_args()
    main(args)