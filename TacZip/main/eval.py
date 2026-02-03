import os
import uuid
import logging
import datasets
from dataclasses import dataclass, field, asdict
from typing import List
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from transformers import HfArgumentParser, set_seed, AutoModel, AutoTokenizer
from tqdm import tqdm
from accelerate import Accelerator
from src.config import dataset2prompt, dataset2compression_instruction, dataset2maxlen, dataset2metric
from src.metric import Metric, answer_cleansing_zero_shot
from src.args import ModelArgs, LoraArgs
from src.utils import save_to_json, move_to_device, FileLogger
from src.data import INPUT_TAG

@dataclass
class TaskArgs:
    cpu: bool = field(
        default=False,
    )
    data_dir: str = field(
        default="data/eval",
    )
    dataset_names: List[str] = field(
        default=lambda: ["hotpotqa", "2wikimqa", "musique"],
    )
    max_length: int = field(
        default=3500,
    )
    comp_ratio: int = field(
        default=16,
    )
    down_scaling_method: str = field(
        default="stride",
    )
    batch_size: int = field(
        default=1,
    )
    output_dir: str = field(
        default="data/results"
    )
    seed: int = field(
        default=42,
    )
    repetition_penalty: float = field(
        default=1.0,
    )
    retrieval_num: int = field(
        default=5,
    )
    compression_type: str = field(
        default="uniform"
    )
    context_proportion: float = field(
        default=0.0625
    ) # the proportion of the importance context in the original context
    low_comp_ratio: int = field(
        default=1
    ) # the compression ratio for important context

    def __post_init__(self):
        if len(self.dataset_names) == 0:
            raise ValueError("`dataset_names` can not be empty.")

def prepare(data_dir: str, dataset_names: List[str], retrieval_num: int):
    def _process(data, dataset_name):
        # * get prompt and replace query placehoder
        prompt = dataset2prompt[dataset_name]
        prompt = prompt.replace(INPUT_TAG, data["input"])

        # * get content
        context = data["context"]
        if dataset_name in ["multi_news", "gov_report", "billsum"]:
            query = "Extract key points from the following documents." + ' </s>'
        else:
            query = data["input"] + ' </s>'

        # * format return
        return {
            "prompt": prompt,
            "context": context,
            "query": query,
        }
    
    def _process_retrieval_results(data: dict, dataset_name: str, retrieval_num: int):
        # * get prompt and replace query placehoder
        prompt = dataset2prompt[dataset_name]
        prompt = prompt.replace(INPUT_TAG, data["query"])

        # * format context
        retrieval_results = data["key"]
        context = "\n\n".join([
            f"Doc {i + 1}: {retrieval_results[i]}"
            for i in range(retrieval_num)
        ])

        # * format return
        return {
            "prompt": prompt,
            "context": context,
            "query": data["query"] + ' </s>'
        }
    
    dataset_dict = datasets.DatasetDict()

    for dataset_name in dataset_names:
        dataset = datasets.load_dataset('json', data_files=os.path.join(data_dir, dataset_name + ".json"), split="train")
        fn_kwargs={"dataset_name": dataset_name}
        if dataset_name in ["nq", "popqa", "trivia", "arc_challenge", "piqa", "social_i_qa"]:
            process_fn = _process_retrieval_results
            fn_kwargs["retrieval_num"] = retrieval_num
        else:
            process_fn = _process

        dataset_dict[dataset_name] = dataset.map(
            process_fn,
            fn_kwargs=fn_kwargs,
            num_proc=32,
            desc=f"prepare {dataset_name}",
        )
        
    return dataset_dict

def collate_fn(batch):
    batch = [
        {k: v for k, v in x.items() if k in ["prompt", "context", "query"]}
        for x in batch
    ]
    return default_collate(batch)

def main():
    from datasets import disable_caching
    disable_caching()
    # * set parser
    parser = HfArgumentParser([ModelArgs, LoraArgs, TaskArgs])
    model_args, lora_args, task_args = parser.parse_args_into_dataclasses()

    # * set logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    
    # * set seed
    set_seed(task_args.seed)

    # * set device
    accelerator = Accelerator(cpu=task_args.cpu)

    # * model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
    )
    model.to(accelerator.device)

    # * load dataset and process
    with accelerator.main_process_first():
        dataset_dict = prepare(
            data_dir=task_args.data_dir,
            dataset_names=task_args.dataset_names,
            retrieval_num=task_args.retrieval_num,
        )

    task_id = str(uuid.uuid4()).replace("-", "")
    metrics_dict = {}
    
    for dataset_name in task_args.dataset_names:
        # Empirically, chat templates tend to be suboptimal for in-context learning tasks.
        if dataset_name in ["trec_fine", "banking77", "clinc150"]:
            apply_chat_template = False
        else:
            apply_chat_template = True

        dataloader = DataLoader(
            dataset_dict[dataset_name],
            batch_size=task_args.batch_size,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        # * generate
        generations = []
        dataloader = accelerator.prepare(dataloader)
        for unprocessed_inputs in tqdm(dataloader, desc=f"eval: {dataset_name}"):
            if task_args.compression_type == "uniform":
                inputs = model.compression_rate_adapter.uniform_allocation(
                    unprocessed_inputs["prompt"],
                    unprocessed_inputs["context"],
                    tokenizer=tokenizer,
                    comp_ratio=task_args.comp_ratio,
                    task_instruction=dataset2compression_instruction[dataset_name],
                    apply_chat_template=apply_chat_template,
                    lm_max_length=model_args.lm_max_length,
                    encoder_max_length=model_args.encoder_max_length,
                    down_scaling_method=task_args.down_scaling_method,
                )
            elif task_args.compression_type == "token_level_adaptation":
                inputs = model.compression_rate_adapter.token_level_adaptation(
                    unprocessed_inputs["prompt"],
                    unprocessed_inputs["context"],
                    tokenizer=tokenizer,
                    queries=unprocessed_inputs["query"],
                    comp_ratio=task_args.comp_ratio,
                    task_instruction=dataset2compression_instruction[dataset_name],
                    apply_chat_template=apply_chat_template,
                    context_proportion=task_args.context_proportion,
                    low_comp_ratio=task_args.low_comp_ratio,
                    lm_max_length=model_args.lm_max_length,
                    encoder_max_length=model_args.encoder_max_length,
                    down_scaling_method=task_args.down_scaling_method,
                )
            inputs = move_to_device(inputs, model.device)
            outputs = model.generate(
                **inputs,
                use_cache=True,
                max_new_tokens=dataset2maxlen[dataset_name],
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                top_p=None,
            )
            outputs = outputs[:, inputs["input_ids"].shape[1]:]

            if accelerator.num_processes > 1:
                outputs = outputs.contiguous()  # must be contiguous
                outputs = accelerator.pad_across_processes(outputs, pad_index=tokenizer.pad_token_id, dim=1)
                outputs = accelerator.gather_for_metrics(outputs)
            
            outputs = outputs.tolist()
            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            if dataset_name in ["arc_challenge", "piqa", "social_i_qa"]:
                outputs = [answer_cleansing_zero_shot(dataset_name, x) for x in outputs]
            
            generations.extend(outputs)

        if accelerator.process_index == 0:
            # * evaluate
            answers = dataset_dict[dataset_name]["answers"]
            metrics = Metric.compute([x.split("\n")[0] for x in generations], answers, dataset2metric[dataset_name], all_classes=dataset_dict[dataset_name]["all_classes"])
            metrics_dict[dataset_name] = metrics
            
            # * save
            questions = dataset_dict[dataset_name]["input"]
            save_to_json(
                os.path.join(task_args.output_dir, task_id, f"{dataset_name}.json"),
                [{"question": question, "output": output, "answers": _answers} for question, output, _answers in zip(questions, generations, answers)],
            )

    # * log metric
    if accelerator.process_index == 0:
        file_logger = FileLogger(os.path.join(task_args.output_dir, "eval.log"))
        file_logger.log(metrics_dict, ModelArgs=asdict(model_args), TaskArgs=asdict(task_args), uuid=task_id)

if __name__ == "__main__":
    main()