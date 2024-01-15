import json
import sys

def read_json_or_jsonl(filename):
    datas = []
    if filename.endswith("jsonl"):
        with open(filename) as f:
            for line in f:
                datas.append(json.loads(line))
    else:
        with open(filename) as f:
            datas = json.load(f)
    return datas


infile,outfile,field = sys.argv[1], sys.argv[2], sys.argv[3]

data = read_json_or_jsonl(infile)
with open(outfile,"w") as fout:
    for i,d in enumerate(data):
        out = d[field].replace("\n"," ")
        fout.write(out + "\n")
