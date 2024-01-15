import json

def read_json_or_jsonl(filename):
    datas = []
    if filename.endswith("jsonl"):
        with open(filename) as f:
            for line in f:
                try:
                    datas.append(json.loads(line))
                except:
                    continue
    else:
        with open(filename) as f:
            datas = json.load(f)
    return datas

def dump_json_or_jsonl(filename,data):
    with open(filename,"w") as fout:
        if filename.endswith("json"):
            json.dump(data,fout,ensure_ascii=False,indent=2)
        elif filename.endswith("jsonl"):
            for d in data:
                fout.write(json.dumps(d,ensure_ascii=False) + "\n")
        else:
            raise ValueError("Unknown file type")