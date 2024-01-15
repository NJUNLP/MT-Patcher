from metrics.base import Scorer
from typing import List
import logging, re


class MismatchScorer(Scorer):
    CHAR_PTN = re.compile(r"([\u2E80-\u9FFF\uA000-\uA4FF\uAC00-\uD7FF\uF900-\uFAFF])")
    
    def __init__(self, source_language, target_language, tokenize=''):
        logging.info(f"{self.__class__.__name__} paramters: trg_lang={target_language}")
        super().__init__(source_language, target_language, tokenize)
    
    
    def compute(self, srcs: List[str], hypos: List[str], refs: List[List[str]]):
        _bad_cnt = 0
        for h in hypos:
            if self._trg_lang == "en":
                if self.CHAR_PTN.search(h):
                    _bad_cnt += 1
            elif self._trg_lang == "zh":
                if not self.CHAR_PTN.search(h):
                    _bad_cnt += 1
            else:
                raise ValueError(f"{self.__class__.__name__} not support target_language={self._trg_lang}")
        return _bad_cnt * 1.0 / len(hypos)