from src.data.sft_dataset import make_sft_data_module
from src.data.seq2seq_dataset import make_seq2seq_data_module

def make_data_module(data_args,model_args,tokenizer,type):
    if type == "sft":
        return make_sft_data_module(data_args,model_args,tokenizer)
    elif type == "seq2seq":
        return make_seq2seq_data_module(data_args,model_args,tokenizer)