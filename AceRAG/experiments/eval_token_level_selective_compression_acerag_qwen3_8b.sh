SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $(dirname $SHELL_FOLDER)

# LMQA, SUM, ICL, Reasoning
torchrun --nproc_per_node 8 -m main.eval \
--model_name_or_path wcyno23/AceRAG-Qwen3-8b \
--window_mode false \
--lm_max_length 3500 \
--encoder_num_hidden_layers 8 \
--encoder_max_length 4096 \
--attn_implementation flash_attention_2 \
--dataset_names hotpotqa 2wikimqa musique multi_news gov_report billsum trec_fine banking77 clinc150 arc_challenge piqa social_i_qa \
--comp_ratio 8 \
--down_scaling_method stride \
--compression_type "token_level_adaptation" \
--low_comp_ratio 1 \
--batch_size 4 \
--use_safetensors True \
--context_proportion 0.0625 \


# ODQA
torchrun --nproc_per_node 8 -m main.eval \
--model_name_or_path wcyno23/AceRAG-Qwen3-8b \
--window_mode false \
--lm_max_length 4096 \
--encoder_num_hidden_layers 8 \
--encoder_max_length 4096 \
--attn_implementation flash_attention_2 \
--dataset_names nq popqa trivia \
--comp_ratio 8 \
--down_scaling_method stride \
--compression_type "token_level_adaptation" \
--low_comp_ratio 1 \
--batch_size 4 \
--use_safetensors True \
--context_proportion 0.0625 \

