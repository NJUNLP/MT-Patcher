from sacrebleu.metrics import BLEU
from metrics.base import Scorer
from typing import List
import logging


class BLEUScorer(Scorer):
    def __init__(self, source_language, target_language, tokenize='13a'):
        if tokenize == '13a':
            if "ja" == target_language or "japan" in target_language:
                tokenize = 'ja-mecab'
            elif "ko" == target_language or "korean" in target_language:
                tokenize = 'ko-mecab'
            if "zh" == target_language or "chinese" in target_language:
                tokenize = 'zh'
        logging.info(f"{self.__class__.__name__} paramters: trg_lang={target_language}, tokenize={tokenize}")
        self.bleu_client = BLEU(trg_lang=target_language, tokenize=tokenize)
        super().__init__(source_language, target_language, tokenize)
    
    def compute(self, srcs: List[str], hypos: List[str], refs: List[List[str]]):
        # logging bleu score
        return self.bleu_client.corpus_score(hypos, refs).score