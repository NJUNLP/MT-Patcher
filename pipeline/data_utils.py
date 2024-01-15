from time import sleep
import re
import json
from tqdm import tqdm
import os
import re
import torch


class FeedbackDataManager:
    output_key = "model_assessment"
    prompt = """假设你是一个非常专业的翻译人员，擅长对机器翻译的结果给出详细完整的评估。 我会给你一句中文句子X和它的英文翻译Y，请你帮忙评估该翻译。
1. 你应该首先给出总体评价。
2. 紧接着，如果有错误，请指出错误，并加以解释。如果没有错误，就不用给出解释。
3. 在解释错误的时候，要求给出错误对应原文片段，错误的理由，以及正确的翻译。
4. 对于多个错误，你应该分条说明。尽量抽取出包含错误的最小片段，并加以说明，避免出现错误位置是整个句子的情况。
5. 你的回答应该是中文。

中文原文: <srctext>
英文译文: <tgttext>
评估: 
"""

    def __init__(self):
        pass

    def read_data(self,infile):
        prompts = []
        datas = []
        with open(infile) as f:
            for line in f:
                line = line.encode("utf-8")
                d = json.loads(line)
                src,tgt = d["src_text"].strip(), d["tgt_text"][0].strip()
                prompt = self.prompt.replace("<srctext>",src).replace("<tgttext>",tgt)
                if len(prompt.split()) > 1024:
                    continue
                prompts.append(self.prompt.replace("<srctext>",src).replace("<tgttext>",tgt))
                datas.append(d)
        return prompts, datas

    def postprocess(self,c):
        ret = c.split("评估:",1)[1].strip()
        print(ret)
        return ret


def make_data_manager(args):
    if args.task == "feedback":
        return FeedbackDataManager()

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

