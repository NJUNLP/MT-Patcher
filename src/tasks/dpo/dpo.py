import torch
from transformers import Trainer
from src.tasks.utils import PeftTrainer, _get_logprob

def get_logprob(model,input_ids,labels):
    output = model(input_ids)
    return _get_logprob(output.logits,labels)

class DPOTask(torch.nn.Module):
    def __init__(self,beta,model_ref) -> None:
        super().__init__()
        self.beta = beta
        self.model_ref = model_ref
        self.log_sigmoid = torch.nn.LogSigmoid()

    def compute_loss(self,model,batch):
        good, good_labels, bad, bad_labels = batch["good_examples"], batch["good_examples_labels"], batch["bad_examples"], batch["bad_examples_labels"]
        policy_good = get_logprob(model,good,good_labels)
        policy_bad = get_logprob(model,bad,bad_labels)
        with torch.no_grad():
            ref_good = get_logprob(self.model_ref,good,good_labels)
            ref_bad = get_logprob(self.model_ref, bad, bad_labels)

        loss = - self.log_sigmoid(self.beta*(policy_good - ref_good - policy_bad + ref_bad))
        return loss


class DPOTrainer(Trainer):
    def __init__(self,beta,add_lm_loss,**kwargs):
        super().__init__(**kwargs)
        self.beta = beta
        self.log_sigmoid = torch.nn.LogSigmoid()
        self.add_lm_loss = add_lm_loss

    def compute_loss(self,model,batch):
        good, good_labels, bad, bad_labels = batch["good_examples"], batch["good_examples_labels"], batch["bad_examples"], batch["bad_examples_labels"]
        policy_good = get_logprob(model,good,good_labels)
        policy_bad = get_logprob(model,bad,bad_labels)

        ref_good, ref_bad = batch["ref_good"], batch["ref_bad"]

        good_reward= policy_good - ref_good
        bad_reward = policy_bad - ref_bad
        # policy_logratios = policy_good - policy_bad
        # ref_logratios = ref_good - ref_bad
        # logits = policy_logratios - ref_logratios
        if not self.add_lm_loss:
            losses = -self.log_sigmoid(good_reward - self.beta * bad_reward)
        else:
            losses = -(self.beta * self.log_sigmoid(good_reward - bad_reward)  + policy_good.sum())

        # good_rewards = (policy_good - ref_good).detach()
        # bad_rewards = (policy_bad - ref_bad).detach()
        # accuracy = (good_rewards > bad_rewards).sum() / good_rewards.size(0)

        # self.log({"Good Reward": good_rewards.mean().item(), "Bad Reward": bad_rewards.mean().item(), "Reward Accuracy": accuracy.item()})

        return losses.sum()

class DPOPeftTrainer(PeftTrainer):
    def __init__(self,beta,add_lm_loss,**kwargs):
        super().__init__(**kwargs)
        self.beta = beta
        self.log_sigmoid = torch.nn.LogSigmoid()
        self.add_lm_loss = add_lm_loss

    def compute_loss(self,model,batch):
        good, good_labels, bad, bad_labels = batch["good_examples"], batch["good_examples_labels"], batch["bad_examples"], batch["bad_examples_labels"]
        policy_good = get_logprob(model,good,good_labels)
        policy_bad = get_logprob(model,bad,bad_labels)

        ref_good, ref_bad = batch["ref_good"], batch["ref_bad"]

        good_reward= policy_good - ref_good
        bad_reward = policy_bad - ref_bad
        if not self.add_lm_loss:
            losses = -self.log_sigmoid(self.beta*good_reward - self.beta * bad_reward)
        else:
            losses = -(self.beta * self.log_sigmoid(good_reward - bad_reward)  + policy_good.sum())

        # good_rewards = (policy_good - ref_good).detach()
        # bad_rewards = (policy_bad - ref_bad).detach()
        # accuracy = (good_rewards > bad_rewards).sum() / good_rewards.size(0)

        # self.log({"Good Reward": good_rewards.mean().item(), "Bad Reward": bad_rewards.mean().item(), "Reward Accuracy": accuracy.item()})

        return losses.sum()



        

    

    

    