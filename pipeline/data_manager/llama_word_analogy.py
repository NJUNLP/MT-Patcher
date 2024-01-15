from time import sleep
import re
import json
from tqdm import tqdm
import os
import re
from pipeline.data_utils import read_json_or_jsonl

class WordAnalogyDataManager:
    beam_size = 1
    output_key = "word_analogy"
    temperature = 1.0
    prompt = """[INST]
<<SYS>>
Assume you are a <srclang>-<tgtlang> language expert with a wealth of knowledge and associative ability in both languages. 
<</SYS>>
I will give you a word/phrase P from an <srclang> sentence X. Please associate from the following aspects and generate three words similar to X for each aspect, and provide the <tgtlang> translation of these words.
Aspects of association:

- Category. What kind of category does this word belong to?
- Semantics. What words often appear in the same context as the given word?

NOTE, the associated words should be rare words, so that it is unlike for a machine translation system to translate it correctly.

Input:
X: <src_text>
P: <error_word>
[/INST]
Output:
"""

    def __init__(self):
        pass

    def read_data(self,infile):
        prompts = []
        datas = read_json_or_jsonl(infile)
        for d in tqdm(datas):
            src = d["src_text"].strip()
            prompt = self.prompt.replace("<src_text>",src)
            prompts.append(prompt)
        return prompts, datas


