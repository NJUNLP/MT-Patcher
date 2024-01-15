import argparse
from tqdm import tqdm
import torch
import argparse
from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
from src.utils import load_model_and_tokenizer, LMPrefixDataLoader, collate_tokens
import sacrebleu

def load_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path,low_cpu_mem_usage=True,).half().cuda()
    model.eval()
    return model, tokenizer

def translate(tokenizer,model,sents,max_tokens,tgtlang):
    dataloader = LMPrefixDataLoader(sents,tokenizer,max_tokens=max_tokens)
    translations = []
    ids = []
    for batch in tqdm(dataloader):
        input = collate_tokens([b[0] for b in batch],pad_idx=tokenizer.pad_token_id).cuda()
        attention_mask = input.ne(tokenizer.pad_token_id)
        encoding = {'input_ids':input, 'attention_mask': attention_mask}
        with torch.no_grad():
            generated_ids = model.generate(
                **encoding,
                max_new_tokens=150,
                num_beams=4,
                early_stopping=True,
                forced_bos_token_id=tokenizer.lang_code_to_id[tgtlang]
                )
        translations.extend(tokenizer.batch_decode(generated_ids,skip_special_tokens=True))
        ids.extend([b[1] for b in batch])
    translations = [e for _,e in sorted(zip(ids,translations),key=lambda pair: pair[0])]
    return translations

def load_test_file(filename,srclang,tgtlang,subset_size):
    with open(filename+"."+srclang) as fsrc, open(filename+"."+tgtlang) as ftgt:
        srclines = [line.strip() for line in fsrc][:subset_size]
        tgtlines = [line.strip() for line in ftgt][:subset_size]
    return srclines,tgtlines

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name-or-path')
    parser.add_argument('--max-tokens',type=int,default=256)
    parser.add_argument('--test-file')
    parser.add_argument('--savedir')
    parser.add_argument('--subset-size',type=int)
    parser.add_argument("--srclang")
    parser.add_argument("--tgtlang")
    parser.add_argument("--tokenize")

    mapping = {
        "en": "eng_Latn",
        "de": "deu_Latn",
        "fr": "fra_Latn",
        "ca": "cat_Latn",
        "fi": "fin_Latn",
        "ru": "rus_Cyrl",
        "bg": "bul_Cyrl",
        "zh": "zho_Hans",
        "ko": "kor_Hang",
        "ar": "arb_Arab",
        "sw": "swh_Latn",
        "hi": "hin_Deva",
        "ta": "tam_Taml"
    }
    tgtlangs = ["en"]
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args)
    print('Evaluating {}-{}'.format(args.srclang,args.tgtlang))
    srcs,tgts = load_test_file(args.test_file,args.srclang,args.tgtlang,subset_size=args.subset_size)
    tokenizer.src_lang = mapping[args.srclang]
    tokenizer.tgt_lang = mapping[args.tgtlang]
    translations = translate(tokenizer,model,srcs,args.max_tokens,tgtlang=mapping[args.tgtlang])
    bleu = sacrebleu.corpus_bleu(translations,[tgts],tokenize=args.tokenize)
    print(bleu)
    with open(args.savedir+"/{}-{}.txt".format(args.srclang,args.tgtlang),'w') as fout:
        for src, ref, hyp in zip(srcs, tgts,translations):
            fout.write(src + "\n" + ref + "\n" + hyp + "\n\n")
