from time import sleep
import re
import json
from tqdm import tqdm
import os
import re
from pipeline.data_utils import read_json_or_jsonl

class SentenceAnalyzerDataManager:
    beam_size = 1
    output_key = "sentence_analysis"
    temperature = 0.2
    prompt = """[INST]
<<SYS>>
Suppose you are a language expert of <srclang> and <tgtlang>.
<</SYS>>
Given a sentence X, please point out its topic, domain and style.
Input:
X: <src_text>
[/INST]
Output:
"""

    def __init__(self):
        pass

    def read_data(self,infile,srclang,tgtlang):
        prompts = []
        datas = read_json_or_jsonl(infile)
        for d in tqdm(datas):
            src = d["src_text"].strip()
            prompt = self.prompt.replace("<src_text>",src).replace(srclang,tgtlang)
            prompts.append(prompt)
        return prompts, datas


