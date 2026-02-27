# Training

The training process consists of two stages:

- Pretrain
  - 90K samples from [redpajama](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T-Sample) with auto-regressive language modeling
  
- Finetune
  - 10K samples from [LongAlpaca](https://huggingface.co/datasets/Yukang/LongAlpaca-12k)
  - 130K samples from HotpotQA, NQ, MulitNews, GovReport, TREC-fine, Banking7, ARC-easy and ARC-challenge.

Please download the training data from [AceRAG-Data]( https://huggingface.co/datasets/wcyno23/AceRAG-Data/tree/main/train/compressive_encoder) and place it under the `data` directory.

## Llama2-7B

### Pretrain

```bash
OUTPUE_NAME=acerag-llama2-pretrain
mkdir -p data/outputs/compressive_encoder/pretrain/${OUTPUE_NAME}

torchrun --nproc_per_node 8 -m main.train_compressive_encoder \
--language_model_name_or_path meta-llama/Llama-2-7b-chat-hf \
--window_mode True \
--min_length 1024 \
--max_length 16000 \
--encoder_name_or_path meta-llama/Llama-2-7b-chat-hf \
--encoder_num_hidden_layers 8 \
--window 1024 \
--encoder_max_length 4096 \
--comp_candidates 1 2 4 8 \
--pretraining_down_scaling_method random \
--data_files data/train/compressive_encoder/redpajama.json \
--output_dir data/outputs/compressive_encoder/pretrain/${OUTPUE_NAME} \
--save_strategy epoch \
--deepspeed data/ds_config/ds_config_stage2.json \
--gradient_checkpointing \
--learning_rate 5e-5 \
--attn_implementation flash_attention_2 \
--per_device_train_batch_size 1
```

### Finetune

```bash
OUTPUE_NAME=acerag-llama2-longalpaca
mkdir -p data/outputs/compressive_encoder/ft/${OUTPUE_NAME}

BASE="data/outputs/compressive_encoder/pretrain/acerag-llama2-pretrain"
LATEST_CKPT_DIR=$(ls -d ${BASE}/checkpoint-* | sort -V | tail -n 1)
ENCODER_PATH="${LATEST_CKPT_DIR}/compressive_encoder"
echo "Using encoder path: $ENCODER_PATH"

torchrun --nproc_per_node 8 -m main.train_compressive_encoder \
--language_model_name_or_path meta-llama/Llama-2-7b-chat-hf \
--window_mode false \
--lm_max_length 4096 \
--encoder_name_or_path "$ENCODER_PATH" \
--encoder_num_hidden_layers 8 \
--encoder_max_length 4096 \
--comp_candidates 1 2 4 8 \
--data_files data/train/compressive_encoder/longalpaca.json \
--min_length 1024 \
--max_length 32000 \
--learning_rate 1e-5 \
--down_scaling_method random \
--output_dir data/outputs/compressive_encoder/ft/${OUTPUE_NAME} \
--save_strategy epoch \
--deepspeed data/ds_config/ds_config_stage2.json \
--stage ft \
--gradient_checkpointing \
--use_safetensors True
```

```bash
OUTPUE_NAME=acerag-llama2-multitask-ft
mkdir -p data/outputs/compressive_encoder/ft/${OUTPUE_NAME}

BASE="data/outputs/compressive_encoder/ft/acerag-llama2-longalpaca"
LATEST_CKPT_DIR=$(ls -d ${BASE}/checkpoint-* | sort -V | tail -n 1)
ENCODER_PATH="${LATEST_CKPT_DIR}/compressive_encoder"
echo "Using encoder path: $ENCODER_PATH"

torchrun --nproc_per_node 8 -m main.train_compressive_encoder \
--language_model_name_or_path meta-llama/Llama-2-7b-chat-hf \
--window_mode false \
--lm_max_length 4096 \
--encoder_name_or_path "$ENCODER_PATH" \
--encoder_num_hidden_layers 8 \
--encoder_max_length 4096 \
--comp_candidates 1 2 4 8 \
--data_files data/train/compressive_encoder/hotpotqa.json data/train/compressive_encoder/nq.json data/train/compressive_encoder/multi_news.json data/train/compressive_encoder/gov_report.json data/train/compressive_encoder/trec_fine.json data/train/compressive_encoder/banking77.json data/train/compressive_encoder/arc_easy.json data/train/compressive_encoder/arc_challenge.json \
--min_length 1024 \
--max_length 32000 \
--learning_rate 1e-5 \
--down_scaling_method random \
--output_dir data/outputs/compressive_encoder/ft/${OUTPUE_NAME} \
--data_splits 0.1 0.1 1.0 1.0 0.1 0.1 0.1 0.1 \
--save_strategy epoch \
--train_num_per_data 50000 \
--deepspeed data/ds_config/ds_config_stage2.json \
--stage ft \
--gradient_checkpointing \
--use_safetensors True
```

## Qwen3-8B

### Pretrain

```bash
OUTPUE_NAME=acerag-qwen3-pretrain
mkdir -p data/outputs/compressive_encoder/${OUTPUE_NAME}

torchrun --nproc_per_node 8 -m main.train_compressive_encoder \
--language_model_name_or_path Qwen/Qwen3-8B \
--window_mode True \
--min_length 1024 \
--max_length 7000 \
--encoder_name_or_path Qwen/Qwen3-8B \
--encoder_num_hidden_layers 8 \
--window 1024 \
--encoder_max_length 4096 \
--comp_candidates 1 2 4 8 \
--pretraining_down_scaling_method random \
--data_files data/train/compressive_encoder/redpajama.json \
--output_dir data/outputs/compressive_encoder/pretrain/${OUTPUE_NAME} \
--save_strategy steps \
--save_steps 0.249999 \
--deepspeed data/ds_config/ds_config_stage2.json \
--gradient_checkpointing \
--learning_rate 5e-5 \
--attn_implementation flash_attention_2 \
--per_device_train_batch_size 1 \
--use_safetensors True
```

### Finetune

```bash
OUTPUE_NAME=acerag-qwen3-longalpaca
mkdir -p data/outputs/compressive_encoder/ft/${OUTPUE_NAME}

BASE="data/outputs/compressive_encoder/pretrain/acerag-qwen3-pretrain"
LATEST_CKPT_DIR=$(ls -d ${BASE}/checkpoint-* | sort -V | tail -n 1)
ENCODER_PATH="${LATEST_CKPT_DIR}/compressive_encoder"
echo "Using encoder path: $ENCODER_PATH"

torchrun --nproc_per_node 8 -m main.train_compressive_encoder \
--language_model_name_or_path Qwen/Qwen3-8B \
--window_mode false \
--lm_max_length 4096 \
--encoder_name_or_path "$ENCODER_PATH" \
--encoder_num_hidden_layers 8 \
--encoder_max_length 4096 \
--comp_candidates 1 2 4 8 \
--data_files data/train/compressive_encoder/longalpaca.json \
--min_length 1024 \
--max_length 32000 \
--learning_rate 1e-5 \
--down_scaling_method random \
--output_dir data/outputs/compressive_encoder/ft/${OUTPUE_NAME} \
--save_strategy epoch \
--deepspeed data/ds_config/ds_config_stage2.json \
--stage ft \
--gradient_checkpointing \
--use_safetensors True
```

```bash
OUTPUE_NAME=acerag-qwen3-multitask-ft
mkdir -p data/outputs/compressive_encoder/ft/${OUTPUE_NAME}

BASE="data/outputs/compressive_encoder/ft/acerag-qwen3-longalpaca"
LATEST_CKPT_DIR=$(ls -d ${BASE}/checkpoint-* | sort -V | tail -n 1)
ENCODER_PATH="${LATEST_CKPT_DIR}/compressive_encoder"
echo "Using encoder path: $ENCODER_PATH"

torchrun --nproc_per_node 8 -m main.train_compressive_encoder \
--language_model_name_or_path Qwen/Qwen3-8B \
--window_mode false \
--lm_max_length 4096 \
--encoder_name_or_path "$ENCODER_PATH" \
--encoder_num_hidden_layers 8 \
--encoder_max_length 4096 \
--comp_candidates 1 2 4 8 \
--data_files data/train/compressive_encoder/hotpotqa.json data/train/compressive_encoder/nq.json data/train/compressive_encoder/multi_news.json data/train/compressive_encoder/gov_report.json data/train/compressive_encoder/trec_fine.json data/train/compressive_encoder/banking77.json data/train/compressive_encoder/arc_easy.json data/train/compressive_encoder/arc_challenge.json \
--min_length 1024 \
--max_length 32000 \
--learning_rate 1e-5 \
--down_scaling_method random \
--output_dir data/outputs/compressive_encoder/ft/${OUTPUE_NAME} \
--data_splits 0.1 0.1 1.0 1.0 0.1 0.1 0.1 0.1 \
--save_strategy epoch \
--train_num_per_data 50000 \
--deepspeed data/ds_config/ds_config_stage2.json \
--stage ft \
--gradient_checkpointing \
--use_safetensors True
```

