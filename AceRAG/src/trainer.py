import os
import torch
from typing import Optional
from transformers.trainer import *


class CompressiveEncoderTrainer(Trainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        self.model.save(output_dir)

        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def _get_train_sampler(self, train_dataset) -> Optional[torch.utils.data.Sampler]:
        if train_dataset is None or not has_length(train_dataset):
            return None
        
        return super()._get_train_sampler()


class BiTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save model & config in safetensors format
        if not hasattr(self.model, "save_pretrained"):
            raise NotImplementedError(
                f"MODEL {self.model.__class__.__name__} "
                f"does not support save_pretrained interface"
            )
        else:
            self.model.save_pretrained(
                output_dir,
                safe_serialization=True
            )
        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss
