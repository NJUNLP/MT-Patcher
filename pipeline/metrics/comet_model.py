import logging

from metrics.base import Scorer
from typing import List
from utils.env import PSM, GPU_NUMS
from comet import load_from_checkpoint


class COMETScorer(Scorer):
    def __init__(self, source_language, target_language, checkpoint, tokenize=''):
        logging.info(f"{self.__class__.__name__} paramters: trg_lang={target_language}, src_lang={source_language}, tokenize={tokenize}, checkpoint={checkpoint}")
        self.comet_client = load_from_checkpoint(checkpoint_path=checkpoint)
        self._batch_size = 8
        self._gpus = int(GPU_NUMS) if PSM is not None else 1 # todo get workspace gpu numbers
        super().__init__(source_language, target_language, tokenize)
        logging.info(f"{self.__class__.__name__} paramters: trg_lang={target_language}, src_lang={source_language}, tokenize={tokenize}, checkpoint={checkpoint}, batch_size={self._batch_size}")
    
    def compute(self, srcs: List[str], hypos: List[str], refs: List[List[str]]):
        score = []
        # logging bleurt score
        for _ref in refs:
            data = [
                { "src": src, "mt": hypo, "ref": ref } for src, hypo, ref in zip(srcs, hypos, _ref)
            ]
            score.extend(self.comet_client.predict(data, self._batch_size, self._gpus)[0])
        return  sum(score) * 1.0 / len(score)