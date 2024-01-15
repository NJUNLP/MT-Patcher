from os import read
import sys
import json
from pipeline.data_utils import read_json_or_jsonl
infile = sys.argv[1]
num_of_chunks = int(sys.argv[2])

datas = read_json_or_jsonl(infile)

print(len(datas))

chunk_size = len(datas) // num_of_chunks
for i in range(num_of_chunks):
    with open("tmp/chunk{}.jsonl".format(i),"w") as fout:
        if i == num_of_chunks - 1:
            sub_datas = datas[(chunk_size*i):]
        else:
            sub_datas = datas[(chunk_size*i):chunk_size*(i+1)]
        for d in sub_datas:
            fout.write(json.dumps(d) + "\n")
