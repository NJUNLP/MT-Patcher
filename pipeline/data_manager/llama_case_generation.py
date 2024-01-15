from time import sleep
import re
import json
from tqdm import tqdm
import os
from copy import deepcopy
import re
from pipeline.data_utils import read_json_or_jsonl


class CaseGenerationDataManager:
    num_case = 4
    beam_size = 1
    output_key = "synthesized_case"
    temperature = 1.0
    prompt = """[INST]
<<SYS>>
Suppose you are a language expert of <srclang> and <tgtlang>.
<<SYS>>
Given a topic, a domain and a style, as well as a bilingual word pair, please generate a pair of parallel sentence that adhere to the given topic, domain and style. They should also contain the given word pair.
Input:
<domain_topic_style>
Word Pair: <word_pair>
[/INST]
Output:
"""

    def __init__(self):
        pass

    def read_data(self,infile,srclang,tgtlang):
        prompts = []
        datas = read_json_or_jsonl(infile)
        ret_datas = []
        for d in datas:
            sentence_analysis = d["sentence_analysis"]
            for error in d["model_assessment_parsed"]:
                P, Q = error["error_source"].strip(), error["correction"].strip()
                new_d = deepcopy(d)
                new_d["error_source"], new_d["correction"] = P,Q
                prompt = self.prompt.replace("<domain_topic_style>",sentence_analysis).replace("<word_pair>","{}({})".format(P,Q)).replace("<srclang>",srclang).replace("<tgtlang>",tgtlang)
                for i in range(self.num_case):
                    prompts.append(prompt)
                    ret_datas.append(new_d)
        return prompts, ret_datas

    def postprocess(self,c):
        return re.split(r"\[/INST\]\nOutput:\n",c)[1]


