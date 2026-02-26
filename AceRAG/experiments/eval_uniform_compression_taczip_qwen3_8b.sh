SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $(dirname $SHELL_FOLDER)

# LMQA, SUM, ICL, Reasoning
torchrun --nproc_per_node 8 -m main.eval \
--model_name_or_path wcyno23/TacZip-Qwen3-8b \
--window_mode false \
--lm_max_length 3500 \
--encoder_num_hidden_layers 8 \
--encoder_max_length 4096 \
--attn_implementation flash_attention_2 \
--dataset_names hotpotqa 2wikimqa musique multi_news gov_report billsum trec_fine banking77 clinc150 arc_challenge piqa social_i_qa \
--comp_ratio 8 \
--down_scaling_method stride \
--batch_size 4 \
--use_safetensors True 

# ODQA
torchrun --nproc_per_node 8 -m main.eval \
--model_name_or_path wcyno23/TacZip-Qwen3-8b \
--window_mode false \
--lm_max_length 4096 \
--encoder_num_hidden_layers 8 \
--encoder_max_length 4096 \
--attn_implementation flash_attention_2 \
--dataset_names nq popqa trivia \
--retrieval_num 5 \
--comp_ratio 8 \
--down_scaling_method stride \
--batch_size 8 \
--use_safetensors True 
