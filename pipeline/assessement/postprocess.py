import json
import re
from pipeline.data_utils import read_json_or_jsonl, dump_json_or_jsonl
import argparse


def parse(d):
    assessment = d["model_assessment"]
    error_sources = re.findall(r"Error\sSource\sWord:(.+?)\n",assessment)
    reasons = re.findall(r"Explanation:(.+?)\n",assessment)
    corrections = re.findall(r"Correction:(.+?)\n",assessment)
    pe = re.findall(r"Better\sTranslation:(.+)",assessment)

    if len(pe) != 1:
        return None,None

    pe = pe[0].strip()

    if len(error_sources) != len(reasons) or len(reasons) != len(corrections):
        return None, None
    else:
        model_assessment_parsed = []
        for es,reason,correction in zip(error_sources,reasons,corrections):
            if correction.strip() in d["tgt_text"]:
                continue
            if correction.strip() not in pe:
                continue
            else:
                model_assessment_parsed.append(
                    {
                        "error_source": es.strip(),
                        "reason": reason.strip(),
                        "correction": correction.strip()
                    }
                )
        if len(model_assessment_parsed) == 0:
            return [], None
        else:
            return model_assessment_parsed, pe

        

def main(args):
    datas = read_json_or_jsonl(args.infile)

    data_with_errors = []
    total = parse_error = invalid = 0
    for d in datas:
        if d["model_assessment"].startswith("No Error."):
            continue
        else:
            total += 1
            model_assessment_parsed, pe = parse(d)
            if model_assessment_parsed is None:
                parse_error += 1
            elif model_assessment_parsed == []:
                invalid += 1
            else:
                d["model_assessment_parsed"] = model_assessment_parsed
                d["pe"] = pe
                data_with_errors.append(d)

    print("Total: {}, Parse Error: {}, Invalid: {}".format(total,parse_error,invalid))
    dump_json_or_jsonl(args.outfile,data_with_errors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile")
    parser.add_argument("--outfile")

    args = parser.parse_args()
    main(args)

    
