import json
import re
from tqdm import tqdm

def read_file(filename):
    datas = []
    if filename.endswith("jsonl"):
        with open(filename) as f:
            for line in f:
                datas.append(json.loads(line))
    else:
        with open(filename) as f:
            datas = json.load(f)


    good_data, bad_data = [], []
    for d in datas:
        if 'output' in d:
            continue
        elif "Error segment in source" not in d["gpt4_assessment"]:
            good_data.append(d)
        else:
            bad_data.append(d)
    return  good_data, bad_data

def detect_codeswitch(sent):
    num_of_characters = len(re.findall(r'[\u4e00-\u9fff]', sent))
    chinese_ratio = num_of_characters / len(sent)
    return chinese_ratio > 0
    
def main(args):
    good_data, bad_data = read_file(args.infile)
    for d in good_data:
        d["corrections"] = []
        d["reasons"] = []

    too_long,  trans = 0,0
    bad_data_correction_source = []

    with open("bad_case.txt","w") as fout:
        for d in tqdm(bad_data):
            error_sources = re.findall(r'Error segment in source:\s*[“"”]?(.+?)["“”]?\n',d["gpt4_assessment"])
            reasons = re.findall(r"Error reason:\s*(.+)",d["gpt4_assessment"])
            corrections = re.findall(r'Correct translation:\s*[“"”]?(.+)["“”]?',d["gpt4_assessment"])
            pes = re.findall(r'Better\sTranslation:(.+)',d["gpt4_assessment"])
            if len(pes) != 1:
                continue
            d["pe"] = pes[0]
            d["error_sources"] = []
            d["corrections"] = []
            d["reasons"] = []
            if len(error_sources) != len(corrections):
                continue
            flag = True
            for error_source, correction,reason in zip(error_sources,corrections,reasons):
                correction = correction.strip()
                reason = reason.strip()
                error_source_length = len(error_source.split()) if args.srclang not in  ["zh","ja"] else len(error_source)
                if error_source_length > 20:
                    flag = False
                    fout.write(error_source + "\n" + json.dumps(d,indent=4,ensure_ascii=False) + "\n\n")
                    too_long += 1
                elif "translat" in correction:
                    flag = False
                    trans += 1
                else:
                    if correction.endswith('"') and not correction.startswith('"'):
                        correction = correction[:-1]
                    d["error_sources"].append(error_source.strip())
                    d["corrections"].append(correction.strip())
                    d["reasons"].append(reason)
                    
            if flag:
                bad_data_correction_source.append(d)

    print(len(bad_data_correction_source),len(bad_data))
    final_data = bad_data_correction_source + good_data
    print("Bad Ratio: {}".format(len(bad_data_correction_source)/len(final_data)))
    with open(args.outfile,"w") as fout:
        json.dump(final_data,fout,indent=4,ensure_ascii=False)
                


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile")
    parser.add_argument("--srclang")
    parser.add_argument("--outfile")

    args = parser.parse_args()
    main(args)