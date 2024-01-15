import torch
from transformers import Trainer

from src.tasks.utils import PeftTrainer


class RewardModelTrainer(Trainer):
    
    def compute_loss(self,model,batch):
        good, bad = batch["good_examples"], batch["bad_examples"]
        good_score = model(good).logits
        bad_score = model(bad).logits

        loss = torch.log(1 + torch.exp(bad_score - good_score))
        return loss

class RewardModelPeftTrainer(PeftTrainer):

    def compute_loss(self,model,batch):
        good, bad = batch["good_examples"], batch["bad_examples"]
        good_score = model(good).logits
        bad_score = model(bad).logits

        loss = torch.log(1 + torch.exp(bad_score - good_score))
        return loss