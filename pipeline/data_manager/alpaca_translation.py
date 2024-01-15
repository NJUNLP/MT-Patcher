import json
import re
from pipeline.data_utils import read_json_or_jsonl


class AlpacaTranslationDataManager:
    beam_size = 1
    output_key = "tgt_text"
    prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
    
### Instruction:
Translate the following sentence into English:
<src_text>

### Response:"""

    def __init__(self):
        pass

    def read_data(self,infile):
        prompts,datas = [],[]
        datas = read_json_or_jsonl(infile)
        for d in datas:
            src= d["src_text"].strip() if "src_text" in d else d["source"]
            if len(src) > 2048:
                continue
            prompts.append(self.prompt.replace("<src_text>",src).replace("<srclang>","Chinese").replace("<tgtlang>","English"))
        return prompts, datas

    def postprocess(self,c):
        return re.findall(r"Output:(.+)",c)[0].strip()


