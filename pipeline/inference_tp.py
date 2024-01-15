from time import sleep
from src.utils import load_model_and_tokenizer, generate_batch, LMPrefixDataLoader, generate_batch_vllm
from evaluation.data_manager import make_data_manager
from evaluation.data_utils import dump_json_or_jsonl
import re
import json
from tqdm import tqdm
import os
import re
import torch
from vllm import LLM, SamplingParams


def generate_chunks_vllm(args,data_manager,prompts,datas):
    if args.config_dir is not None and args.config_dir.startswith("hdfs"):
        tokenizer_path = config_path = os.path.basename(args.config_dir)
    else:
        tokenizer_path = args.config_dir
        config_path = args.config_dir

    llm = LLM(model=args.model_path,tokenizer=tokenizer_path,dtype="half",trust_remote_code=True,tensor_parallel_size=8)
    # dataloader = LMPrefixDataLoader(prompts,tokenizer,max_tokens=args.max_tokens)
    
    print("Generating Assessment")
    sampling_params = SamplingParams(
        n=(data_manager.beam_size),
        temperature=data_manager.temperature if hasattr(data_manager,"temperature") else args.temperature,
        top_p=1.0,
        use_beam_search=(data_manager.beam_size>1),
        ignore_eos=False,
        max_tokens=256
    )

    with torch.no_grad():
        outputs = llm.generate(
            prompts,
            sampling_params
        )

    results = [data_manager.postprocess(output.outputs[0].text) for output in outputs]

    return_datas = []
    for r,d in zip(results,datas):
        if hasattr(data_manager,"output_type") and data_manager.output_type == "list":
            if data_manager.output_key in d:
                d[data_manager.output_key].append(r)
            else:
                d[data_manager.output_key] = [r]
        else:
            d[data_manager.output_key] = r
        return_datas.append(d)

    return return_datas

def main(args):

    data_manager = make_data_manager(args)
    prompts, datas = data_manager.read_data(args.infile)
    final_data = generate_chunks_vllm(args,data_manager,prompts,datas)
    dump_json_or_jsonl(args.savefile,final_data)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile")
    parser.add_argument("--savefile")
    parser.add_argument("--task")
    parser.add_argument("--model-path")
    parser.add_argument("--config-dir")
    parser.add_argument("--max-tokens",type=int,default=512)
    parser.add_argument("--use-vllm",default=False,action="store_true")
    parser.add_argument("--devices")
    parser.add_argument("--temperature",type=float,default=0)
    parser.add_argument("--output-key")
    args = parser.parse_args()
    main(args)



