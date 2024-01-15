from time import sleep
import re
import json
from tqdm import tqdm
import os
import re


class FeedbackDataManager:
    temperature = 0.1
    beam_size = 1
    output_key = "model_assessment"
    prompt = """[INST]
<<SYS>>
Assuming you are a highly proficient translator skilled at providing detailed and comprehensive assessments of machine translations. 
<</SYS>>
I will give you a <srclang> sentence X and its <tgtlang> translation Y, and I would like you to help assess the translation.
1. You should first provide an overall assessment.
2. Following that, 
    - If there are no errors, just say "No error." and do not provide an explanation.
    - If there are errors, please specify 
        - the error type, 
        - the corresponding segment in the <srclang> sentence X, 
        - the corresponding segment in the translation Y, 
        - the reason for the error, 
        - and the correct translation for the segment
    - If there are errors, you should also provide a good translation at the end of the assessment.
4. For multiple errors, you should address them separately.
5. [Important!] Try to pinpoint the smallest segments containing errors and explain them, avoiding cases where the error encompasses the entire sentence.
6. Carefully read the original text and the translation to identify all translation errors.
7. For errors related to articles, singular/plural, and similar issues that do not significantly affect semantic comprehension, there is no need to point them out.
8. Your response should be in English.
9. Be concise.

Now, please assess the following translation:

<srclang>: <srctext>
<tgtlang>: <tgttext>
[/INST]
Assessment:
"""
    chinese_characters = re.compile(r'[\u4e00-\u9fff]+')
    words = re.compile(r"\w+")

    def __init__(self):
        pass

    def read_data(self,infile,srclang,tgtlang):
        prompts = []
        datas = []
        filtered = 0
        with open(infile) as f:
            for line in tqdm(f):
                line = line.encode("utf-8")
                d = json.loads(line)
                src,tgt = d["src_text"].strip(), d["tgt_text"].strip()
                if len(self.words.findall(src)) > 256 or len(self.words.findall(tgt)) > 256:
                    filtered += 1
                    continue
                if len(self.chinese_characters.findall(src)) > 256 or len(self.chinese_characters.findall(tgt)) > 256:
                    filtered += 1
                    continue
                prompt = self.prompt.replace("<srctext>",src).replace("<tgttext>",tgt).replace("<srclang>",srclang).replace("<tgtlang>",tgtlang)
                prompts.append(prompt)
                datas.append(d)
        return prompts, datas

    def postprocess(self,c):
        return c




