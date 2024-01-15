from transformers import TrainerCallback
from transformers import Trainer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import torch

from glob import glob
import os
import shutil

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union


TRAINING_ARGS_NAME = "training_args.bin"


class PeftTrainer(Trainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        self.model.save_pretrained(output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args,
        state,
        control,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )
        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if args.local_rank == 0:
            if os.path.exists(pytorch_model_path):
                os.remove(pytorch_model_path)
        return control

class RemoveDeepspeedCheckpointCallback(TrainerCallback):
    def on_save(
        self,
        args,
        state,
        control,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )

        if args.local_rank == 0:
            for deepspeed_dir in glob(checkpoint_folder+"/global_step*"):
                shutil.rmtree(deepspeed_dir)
        return control


def _get_logprob(logits,labels,average_log_prob=False):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss_mask = shift_labels.ne(-100)

    shift_labels[shift_labels.eq(-100)] = 0

    per_token_logps = torch.gather(shift_logits.log_softmax(-1), dim=2, index=shift_labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)
