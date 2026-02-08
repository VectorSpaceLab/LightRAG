import logging
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, set_seed
from src.data import SequenceLevelEmbedCollator, TokenLevelEmbedCollator, SequenceLevelTrainDataset, TokenLevelTrainDataset
from src.trainer import BiTrainer
from src.modeling_taczip import BiEncoderModel
from src.data import CONTEXT_TAG
from src.args import LoraArgs, ModelArgs, TrainingArgs

logger = logging.getLogger(__name__)

@dataclass
class TaskArgs:
    data_files: List[str] = field(
        default_factory=lambda: []
    )
    query_max_len: int = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    passage_max_len: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    train_group_size: int = field(
        default=8
    )
    sub_batch_size: int = field(
        default=None
    )
    train_num_per_data: Optional[int] = field(
        default=None,
    )


def compute_metrics(eval_preds):
    predictions = eval_preds.predictions.argmax(axis=-1)
    labels = eval_preds.label_ids
    accuracy = (predictions == labels).mean()

    print(f"Accuracy: {accuracy}")
    return {"accuracy": accuracy}


def main():
    from datasets import disable_caching
    disable_caching()
    
    parser = HfArgumentParser((ModelArgs, TaskArgs, TrainingArgs, LoraArgs))
    model_args, task_args, training_args, lora_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Task parameters %s", task_args)
    logger.info("Lora parameters %s", lora_args)

    # Set seed
    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.language_model_name_or_path,
    )
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "left"
    tokenizer.add_tokens([CONTEXT_TAG], special_tokens=True)
    
    model = BiEncoderModel(
        model_name=model_args.embedding_model_name_or_path,
        peft_model_name=lora_args.peft_model_name_or_path,
        normlized=training_args.normlized,
        pooling_method=training_args.pooling_method,
        negatives_cross_device=training_args.negatives_cross_device,
        temperature=training_args.temperature,
        lora_tune=model_args.lora_tune,
        enable_token_level_retrieval=training_args.enable_token_level_retrieval,
        attn_implementation=model_args.attn_implementation,
    )

    if training_args.fix_position_embedding:
        for k, v in model.named_parameters():
            if "position_embeddings" in k:
                logging.info(f"Freeze the parameters for {k}")
                v.requires_grad = False

    if training_args.enable_token_level_retrieval:
        train_dataset = TokenLevelTrainDataset(
            data_files=task_args.data_files,
        )
        data_collator = TokenLevelEmbedCollator(
            tokenizer=tokenizer,
            query_max_len=task_args.query_max_len,
        )
    else:
        train_dataset = SequenceLevelTrainDataset(
            data_files=task_args.data_files,
            train_group_size=task_args.train_group_size,
            query_max_len=task_args.query_max_len,
            passage_max_len=task_args.passage_max_len,
            tokenizer=tokenizer,
        )
        data_collator = SequenceLevelEmbedCollator(
            tokenizer=tokenizer,
            query_max_len=task_args.query_max_len,
            passage_max_len=task_args.passage_max_len,
            sub_batch_size=task_args.sub_batch_size,
        )
    
    model.model.config.eos_token_id = tokenizer.eos_token_id
    
    trainer = BiTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    # Training
    if model_args.resume_from_ckpt is not None:
        trainer.train(resume_from_checkpoint=model_args.resume_from_ckpt)
    else:
        trainer.train()

    trainer.save_model()
    # For convenience, we also re-save the tokenizer to the same directory
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)

    
if __name__ == "__main__":
    main()
