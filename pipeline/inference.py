from time import sleep
from src.utils import load_model_and_tokenizer, generate_batch, LMPrefixDataLoader, generate_batch_vllm
from pipeline.data_manager import make_data_manager
from pipeline.data_utils import dump_json_or_jsonl
import re
import json
from tqdm import tqdm
import os
import re
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from copy import deepcopy


def generate_chunks(args,data_manager,prompts,datas,device_id,return_dict):
    sleep(device_id*30)
    if args.config_dir is not None and args.config_dir.startswith("hdfs"):
        tokenizer_path = config_path = os.path.basename(args.config_dir)
    else:
        tokenizer_path = args.config_dir
        config_path = args.config_dir
    tokenizer, model = load_model_and_tokenizer(args.model_path,tokenizer_path,config_path,device="cuda:{}".format(device_id))
    dataloader = LMPrefixDataLoader(prompts,tokenizer,max_tokens=args.max_tokens)
    
    print("Generating Assessment")
    results,ids = [],[]

    for i,batch in enumerate(tqdm(dataloader)):
        completions = generate_batch(model,
            tokenizer,
            batch,
            device="cuda:{}".format(device_id),
            left_pad=True,
            max_new_tokens=256,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=(data_manager.temperature>=0),
            temperature=data_manager.temperature,
            top_p=0.9,
            early_stopping=True)
        ids.extend([b[1] for b in batch])
        pp = [data_manager.postprocess(c) for c in completions]
        results.extend(
            pp
        )

    id2results = {}
    for id, result in zip(ids,results):
        id2results[id] = result

    return_datas = []
    output_key = args.output_key if args.output_key is not None else data_manager.output_key
    for i,d in enumerate(datas):
        if i in id2results:
            _d = deepcopy(d)
            _d[output_key] = id2results[i]
            return_datas.append(_d)
        else:
            continue

    return_dict[device_id] = return_datas


def generate_chunks_vllm(args,data_manager,prompts,datas,device_id,return_dict):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    sleep(device_id*10)

    llm = LLM(model=args.model_path,tokenizer=args.tokenizer_path,dtype="half",trust_remote_code=True)
    
    print("Generating Assessment")
    sampling_params = SamplingParams(
        n=(data_manager.beam_size),
        temperature=data_manager.temperature if hasattr(data_manager,"temperature") else args.temperature,
        ignore_eos=False,
        max_tokens=256
    )

    with torch.no_grad():
        outputs = llm.generate(
            prompts,
            sampling_params
        )

    results = [output.outputs[0].text for output in outputs]

    output_key = args.output_key if args.output_key is not None else data_manager.output_key
    return_datas = []
    for r,_d in zip(results,datas):
        d = deepcopy(_d)
        if hasattr(data_manager,"output_type") and data_manager.output_type == "list":
            if output_key in d:
                d[output_key].append(r)
            else:
                d[output_key] = [r]
        else:
            d[output_key] = r
        return_datas.append(d)

    return_dict[device_id] = return_datas


def main(args):
    process_fn = generate_chunks_vllm if args.use_vllm else generate_chunks
    data_manager = make_data_manager(args)
    
    prompts, datas = data_manager.read_data(args.infile,args.srclang,args.tgtlang)

    devices = [int(i) for i in args.devices.split(",")]
    if len(devices) == 1:
        return_dict = {}
        process_fn(args,data_manager,prompts,datas,0,return_dict)
    else:
        chunk_size = len(prompts) // len(devices)
        torch.multiprocessing.set_start_method("spawn")
        manager = torch.multiprocessing.Manager()
        return_dict = manager.dict()
        jobs = []
        for i, device_id in enumerate(devices):
            start, end = chunk_size*i, min(chunk_size*(i+1),len(prompts))
            sub_prompts, sub_datas = prompts[start:end], datas[start:end]
            p = torch.multiprocessing.Process(target=process_fn, args=(args,data_manager,sub_prompts,sub_datas,device_id,return_dict))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()

    final_data = []
    for data in return_dict.values():
        final_data += data

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
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--srclang")
    parser.add_argument("--tgtlang")
    args = parser.parse_args()
    main(args)



