import json
import re
from pipeline.data_utils import read_json_or_jsonl


class TranslationDataManager:
    beam_size = 1
    output_key = "tgt_text"
    prompt = """Translate the following sentences from <srclang> to <tgtlang>.
Input: <src_text>
Output: """

    def __init__(self):
        pass

    def read_data(self,infile,srclang,tgtlang):
        prompts,datas = [],[]
        datas = read_json_or_jsonl(infile)
        for d in datas:
            src= d["src_text"].strip() if "src_text" in d else d["source"]
            if len(src) > 2048:
                continue
            prompts.append(self.prompt.replace("<src_text>",src).replace("<srclang>",srclang).replace("<tgtlang>",tgtlang))
        return prompts, datas

    def postprocess(self,c):
        return re.findall(r"Output:(.+)",c)[0].strip()


