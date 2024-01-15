from bleurt import score
from metrics.base import Scorer
from typing import List
import logging
import tensorflow as tf


class BLEURTScorer(Scorer):
    def __init__(self, source_language, target_language, checkpoint="", tokenize=''):
        # for low leval tensorflow, we must limit GPU memory
        # https://github.com/tensorflow/tensorflow/issues/45068
        physical_devices = tf.config.list_physical_devices('GPU') 
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        self.bleurt_client = score.BleurtScorer(checkpoint=checkpoint)
        self._batch_size = 8
        super().__init__(source_language, target_language, tokenize)
        logging.info(f"{self.__class__.__name__} paramters: trg_lang={target_language}, src_lang={source_language}, tokenize={tokenize}, checkpoint={checkpoint}, batch_size={self._batch_size}")
    
    def compute(self, srcs: List[str], hypos: List[str], refs: List[List[str]]):
        score = []
        # logging bleurt score
        for _ref in refs:
            score.extend(self.bleurt_client.score(references=_ref, candidates=hypos, batch_size=self._batch_size))
        return  sum(score) * 1.0 / len(score)
        