import torch
from transformers import Trainer

from src.tasks.utils import PeftTrainer

class BehaviorClone(torch.nn.Module):
    
    def compute_loss(self,model,batch):
        teacher_examples, teacher_examples_labels = batch["teacher_examples"], batch["teacher_examples_labels"]
        output = model(teacher_examples,labels=teacher_examples_labels)
        return output.loss


class ConditionalBehaviorClone(torch.nn.Module):

    def compute_loss(self,model,batch):
        good, good_labels, bad, bad_labels = batch["good_examples"], batch["good_examples_labels"], batch["bad_examples"], batch["bad_examples_labels"]
        good_loss = model(good,labels=good_labels)
        bad_loss = model(bad,labels=bad_labels)
        return (good_loss + bad_loss) / 2

    
class ConditionalBehaviorCloneTrainer(Trainer):


    def compute_loss(self,model,batch):
        good, good_labels, bad, bad_labels = batch["good_examples"], batch["good_examples_labels"], batch["bad_examples"], batch["bad_examples_labels"]
        good_loss = model(good,labels=good_labels).loss
        bad_loss = model(bad,labels=bad_labels).loss
        return (good_loss + bad_loss) / 2


class BehaviorCloneTrainer(Trainer):
    
    def compute_loss(self,model,batch):
        teacher_examples, teacher_examples_labels = batch["good_examples"], batch["good_examples_labels"]
        output = model(teacher_examples,labels=teacher_examples_labels)
        return output.loss

class ConditionalBehaviorClonePeftTrainer(PeftTrainer):


    def compute_loss(self,model,batch):
        good, good_labels, bad, bad_labels = batch["good_examples"], batch["good_examples_labels"], batch["bad_examples"], batch["bad_examples_labels"]
        good_loss = model(good,labels=good_labels).loss
        bad_loss = model(bad,labels=bad_labels).loss
        return (good_loss + bad_loss) / 2


class BehaviorClonePeftTrainer(PeftTrainer):
    
    def compute_loss(self,model,batch):
        teacher_examples, teacher_examples_labels = batch["good_examples"], batch["good_examples_labels"]
        output = model(teacher_examples,labels=teacher_examples_labels)
        return output.loss