import torch
import datasets
from datasets import concatenate_datasets
from dataclasses import dataclass, field
from typing import List, Optional
from transformers import HfArgumentParser, set_seed
from src import load_model_and_tokenizer
from src.args import ModelArgs, TrainingArgs
from src.data import Data, DefaultDataCollator, SFTDataCollator, INPUT_TAG, CONTEXT_TAG
from src.trainer import ContextCompressorTrainer
from src.utils import extract_file_name
from src.config import dataset2compression_instruction


@dataclass
class TaskArgs:
    stage: str = field(
        default="pt", metadata={"help": "Training stage: pt (pretrain) or ft (finetune)"}
    )
    data_files: List[str] = field(
        default_factory=lambda: []
    )
    min_length: Optional[int] = field(
        default=None,
    )
    max_length: Optional[int] = field(
        default=None,
    )
    train_num_per_data: Optional[int] = field(
        default=None,
    )
    down_scaling_method: Optional[str] = field(
        default="stride",
    )
    # multi task tuning
    data_splits: List[float] = field(
        default_factory=list,
    )


def resample_dataset(dataset, data_splits, train_num_per_data, seed, idx):
    """
    Determine the target sample size based on data_splits * train_num_per_data,
    then perform downsampling or upsampling on the dataset accordingly.
    
    Args:
        dataset: Original dataset
        data_splits: List of dataset split ratios
        train_num_per_data: Number of samples per dataset
        seed: Random seed
        idx: Current dataset index
    
    Returns:
        Resampled dataset
    """
    if train_num_per_data is None:
        return dataset

    # 1. Determine target sample size
    if len(data_splits) > 0 and idx < len(data_splits):
        target_num = int(train_num_per_data * data_splits[idx])
    else:
        target_num = train_num_per_data
    
    current_num = len(dataset)
    
    # 2. Perform downsampling or upsampling based on target sample size
    if current_num > target_num:
        # Downsampling: current sample size is larger than target
        dataset = dataset.train_test_split(
            test_size=target_num, 
            seed=seed
        )["test"]
    elif current_num < target_num and len(data_splits) > 0:
        # Upsampling: current sample size is smaller than target
        repeat_times = target_num // current_num
        remainder = target_num % current_num
        
        # Repeat the entire dataset
        repeated_dataset = dataset
        for _ in range(repeat_times - 1):
            repeated_dataset = concatenate_datasets([repeated_dataset, dataset])
        
        # Add remaining samples
        if remainder > 0:
            extra_samples = dataset.shuffle(seed=seed).select(range(remainder))
            repeated_dataset = concatenate_datasets([repeated_dataset, extra_samples])
        
        dataset = repeated_dataset
    
    return dataset


def prepare_pretrain_data(data_files):
    dataset_dict = datasets.DatasetDict()

    for data_file in data_files:
        dataset_name = extract_file_name(data_file)
        dataset = datasets.load_dataset(
            "json",
            data_files=data_file,
            split="train",
        )
        dataset_dict[dataset_name] = dataset

    return dataset_dict


def prepare_ft_data(data_files, data_splits, train_num_per_data, seed):
    def _process(data: dict, retrieval_num: int):
        prompt = f"Answer the question based on the given documents. Only give me the answer and do not output any other words.\n\nThe following are given documents.\n{CONTEXT_TAG}\n\nAnswer the question based on the given documents. Only give me the answer and do not output any other words.\n\nQuestion: {INPUT_TAG}\nAnswer:"
        prompt = prompt.replace(INPUT_TAG, data["query"])
        retrieval_results = data["key"]
        context = "\n\n".join([f"Doc {i + 1}: {retrieval_results[i]}" for i in range(min(len(retrieval_results), retrieval_num))])
        content = prompt.replace(CONTEXT_TAG, context)
        return {
            "conversations": [
                {"content": content, "role": "user", "prompt": prompt, "context": context},
                {"content": data["answers"][0], "role": "assistant"},
            ],
        }

    dataset_dict = datasets.DatasetDict()
    for idx, data_file in enumerate(data_files):
        dataset_name = extract_file_name(data_file)
        dataset = datasets.load_dataset("json", data_files=data_file, split="train")
        dataset = resample_dataset(dataset, data_splits, train_num_per_data, seed, idx)
        if "key" in dataset.column_names:
            dataset = dataset.map(_process, fn_kwargs={"retrieval_num": 5}, num_proc=1, desc=f"prepare {dataset_name}", remove_columns=dataset.column_names)
        dataset_dict[dataset_name] = dataset

    print('dataset_dict', dataset_dict)
    return dataset_dict


def main():
    from datasets import disable_caching
    disable_caching()
    torch.cuda.empty_cache()
    # * set parser
    parser = HfArgumentParser([ModelArgs, TaskArgs, TrainingArgs])
    model_args, task_args, training_args = parser.parse_args_into_dataclasses()
    
    # * set seed
    set_seed(training_args.seed)

    # * model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_args)

    # * load dataset
    with training_args.main_process_first():
        if task_args.stage == "pt":
            dataset_dict = prepare_pretrain_data(task_args.data_files)
            for dataset_name in dataset_dict:
                dataset = dataset_dict[dataset_name]
                dataset = dataset.map(
                    Data.encode_pretraining_data,
                    fn_kwargs={
                        "tokenizer": tokenizer,
                        "min_length": task_args.min_length,
                        "max_length": task_args.max_length,
                    },
                    batched=True,
                    num_proc=32,
                    remove_columns=dataset.column_names,
                    batch_size=32,
                    with_indices=True,
                )
                dataset_dict[dataset_name] = dataset
        elif task_args.stage == "ft":
            dataset_dict = prepare_ft_data(task_args.data_files, task_args.data_splits, task_args.train_num_per_data, training_args.seed)
            for dataset_name in dataset_dict:
                if dataset_name in dataset2compression_instruction.keys():
                    task_instruction = dataset2compression_instruction[dataset_name]
                else:
                    task_instruction = None
                dataset = dataset_dict[dataset_name]
                dataset = dataset.map(
                    Data.encode_instruction_tuning_data,
                    fn_kwargs={
                        "tokenizer": tokenizer,
                        "lm_max_length": model_args.lm_max_length,
                        "encoder_max_length": model_args.encoder_max_length,
                        "comp_candidates": model_args.comp_candidates,
                        "down_scaling_method": task_args.down_scaling_method,
                        "task_instruction": task_instruction,
                        "min_length": task_args.min_length,
                        "max_length": task_args.max_length,
                    },
                    batched=True,
                    num_proc=32,
                    remove_columns=dataset.column_names,
                    batch_size=32,
                    with_indices=True,
                )
                dataset_dict[dataset_name] = dataset
        dataset = datasets.concatenate_datasets(dataset_dict.values())

    # * set trainer
    if model_args.window_mode:
        collator = DefaultDataCollator(tokenizer)
    else:
        collator = SFTDataCollator(tokenizer)

    trainer = ContextCompressorTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )
    model.accelerator = trainer.accelerator

    # * training
    trainer.train()

if __name__ == "__main__":
    main()