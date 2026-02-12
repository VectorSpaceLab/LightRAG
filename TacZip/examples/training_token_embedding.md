# Training

The training process consists of two stages:

* Pretrain (Passage Retrieval)
  * 486K samples from [bge-multilingual-gemma2-data](https://huggingface.co/datasets/hanhainebula/bge-multilingual-gemma2-data/tree/main/en/MSMARCO).

* Finetune (Token-level Retrieval)
  * 12K samples from HotpotQA, NQ, MulitNews, GovReport, and ARC-challenge. Important tokens were selected using GPT-4o.

Please download the training data from [TacZip-Data](https://huggingface.co/datasets/wcyno23/TacZip-Data/tree/main/train/token_embedding) and [bge-multilingual-gemma2-data](https://huggingface.co/datasets/hanhainebula/bge-multilingual-gemma2-data/tree/main/en/MSMARCO), and place it under the `data` directory.

## Llama2-7B

The base model is [Yukang/Llama-2-7b-longlora-32k-ft](https://huggingface.co/Yukang/Llama-2-7b-longlora-32k-ft), an extended-context variant of Llama 2.

### Pretrain

```bash
OUTPUE_NAME=llama2-pretrain-msmarco
mkdir -p data/outputs/token_embedding/pretrain/${OUTPUE_NAME}

torchrun --nproc_per_node 8 -m main.train_token_embedding \
--output_dir data/outputs/token_embedding/pretrain/${OUTPUE_NAME} \
--embedding_model_name_or_path "Yukang/Llama-2-7b-longlora-32k-ft" \
--tokenizer_name "meta-llama/Llama-2-7b-chat-hf" \
--data_files data/train/token_embedding/msmarco_hn_train.jsonl \
--lora_tune True \
--learning_rate 2e-4 \
--num_train_epochs 1 \
--gradient_accumulation_steps 1 \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 1 \
--normalized True \
--enable_token_level_retrieval False \
--temperature 0.02 \
--query_max_len 512 \
--passage_max_len 512 \
--train_group_size 8 \
--dataloader_drop_last True \
--gradient_checkpointing \
--bf16 \
--logging_steps 10 \
--save_strategy epoch \
--warmup_ratio 0.1 \
--negatives_cross_device True \
--deepspeed data/ds_config/ds_config_stage1.json \
```

### Finetune

```bash
OUTPUE_NAME=llama2-token-retrieval
mkdir -p data/outputs/token_embedding/ft/${OUTPUE_NAME}

BASE="data/outputs/token_embedding/pretrain/llama2-pretrain-msmarco"
LATEST_CKPT_DIR=$(ls -d ${BASE}/checkpoint-* | sort -V | tail -n 1)
PEFT_MODEL_PATH="${LATEST_CKPT_DIR}"

torchrun --nproc_per_node 8 -m main.train_token_embedding \
--output_dir data/outputs/token_embedding/ft/${OUTPUE_NAME} \
--embedding_model_name_or_path "Yukang/Llama-2-7b-longlora-32k-ft" \
--peft_model_name_or_path "$PEFT_MODEL_PATH" \
--tokenizer_name "meta-llama/Llama-2-7b-chat-hf" \
--data_files data/train/token_embedding/nq_llama2.json data/train/token_embedding/hotpotqa_llama2.json data/train/token_embedding/arc_challenge_llama2.json data/train/token_embedding/gov_report_llama2.json data/train/token_embedding/multi_news_llama2.json \
--lora_tune True \
--learning_rate 2e-4 \
--num_train_epochs 1 \
--gradient_accumulation_steps 8 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--normalized True \
--enable_token_level_retrieval True \
--dataloader_drop_last False \
--temperature 0.02 \
--query_max_len 512 \
--passage_max_len 80000 \
--gradient_checkpointing \
--bf16 \
--logging_steps 5 \
--save_strategy epoch \
--warmup_ratio 0.1 \
--deepspeed data/ds_config/ds_config_stage1.json \
```

## Qwen3-8B

The base model is [Qwen/Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B). As this model has been trained on large-scale retrieval tasks, we omit further pretraining.

### Finetune

```bash
OUTPUE_NAME=qwen3-token-retrieval
mkdir -p data/outputs/token_embedding/ft/${OUTPUE_NAME}

torchrun --nproc_per_node 8 -m main.train_token_embedding \
--output_dir data/outputs/token_embedding/ft/${OUTPUE_NAME} \
--embedding_model_name_or_path "Qwen/Qwen3-Embedding-8B" \
--tokenizer_name "Qwen/Qwen3-8B" \
--data_files data/train/token_embedding/nq_qwen3.json data/train/token_embedding/hotpotqa_qwen3.json  data/train/token_embedding/arc_challenge_qwen3.json data/train/token_embedding/gov_report_qwen3.json data/train/token_embedding/multi_news_qwen3.json \
--lora_tune True \
--learning_rate 2e-4 \
--num_train_epochs 1 \
--gradient_accumulation_steps 8 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--normalized True \
--enable_token_level_retrieval True \
--dataloader_drop_last False \
--temperature 0.02 \
--query_max_len 512 \
--passage_max_len 80000 \
--gradient_checkpointing \
--bf16 \
--logging_steps 5 \
--save_strategy epoch \
--warmup_ratio 0.1 \
--deepspeed data/ds_config/ds_config_stage1.json \
```

