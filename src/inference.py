import readline
import sys
import json

from transformers import AutoTokenizer, AutoConfig
from xenon_generation.models.modeling_gpt2_lab import GPT2LabConfig, GPT2LabLMHeadModel
from src.models.gpt2_lab_flash_attn import GPTLMHeadModel

AutoConfig.register("gpt2lab", GPT2LabConfig)

"""
# auto register
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from modeling_gpt2_lab import GPT2LabConfig, GPT2LabLMHeadModel

AutoConfig.register("gpt2lab", GPT2LabConfig)
AutoModelForCausalLM.register(GPT2LabConfig, GPT2LabLMHeadModel)

model = AutoModelForCausalLM.from_pretrained(path).cuda()
"""

# 上文训练参数中 --trainer.default_hdfs_dir 指定的路径
model_path = sys.argv[1]
# 上文训练参数中 --data.tokenizer 指定的路径
tokenizer_path = sys.argv[1]

data_file = sys.argv[2]

save_file = sys.argv[3]

with open(data_file) as f:
    sources = [json.loads(line)["src_text"] for line in f.readlines()]

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
config = AutoConfig.from_pretrained(model_path)
model = GPTLMHeadModel.from_pretrained(model_path,config=config).half().cuda()

model.eval()

def generate(text):
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    inputs.pop("token_type_ids")  # 注意！如果不删除，会导致每个token加上一个type emb 
    res = model.generate(
        inputs["input_ids"], max_length=512, temperature=0.5
    )
    return tokenizer.decode(res[0])


targets = []
for src in sources:
    target = generate(
        "{}[PLHD95] 请翻译成英文. ".format(src)
    )
    targets.append(target)

results = [
    {"src": src, "tgt": tgt} for src,tgt in zip(sources,targets)
]

with open(save_file,"w") as f:
    json.dump(results,f,ensure_ascii=False,indent=2)
