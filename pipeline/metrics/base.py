from typing import List
from abc import ABCMeta



class Scorer(metaclass=ABCMeta):
    def __init__(self, source_language, target_language, tokenize=''):
        self._src_lang = source_language
        self._trg_lang = target_language
        self._tokenize = tokenize
        super().__init__()
    
    def compute(self, srcs: List[str], hypos: List[str], refs: List[List[str]]):
        raise NotImplementedError