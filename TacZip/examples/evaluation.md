# Evaluation

TacZip is benchmarked across five task categories: Long-sequence Multi-document QA(**LMQA**), Open-Domain QA(**ODQA**), Summarization(**SUM**), In-Context Learning(**ICL**) and Commonsense Reasoning(**Reasoning**).

- **LMQA**: HotpotQA, 2WikiMQA and Musique

- **ODQA**: NQ, PopQA and TriviaQA

- **SUM**: MultiNews, GovReport and BillSum

- **ICL**: Trec-fine, Banking77 and Clinc150

- **Reasoning**: ARC-challenge, PIQA and Social IQa

Download the evaluation dataset from the Hugging Face repository:
https://huggingface.co/datasets/wcyno23/TacZip-Data/tree/main/eval
and place it in the `data/eval` directory.

All experiment scripts are available in the `experiments` directory. The evaluation results will be saved under `data/results`.

## Uniform Compression/Allocation
#### TacZip-Qwen3-8b

```bash
bash experiments/eval_uniform_compression_taczip_qwen3_8b.sh
```


#### TacZip-Llama2-7b

```bash
bash experiments/eval_uniform_compression_taczip_llama2_7b.sh
```


## Token-level Ratio Adaptation

#### TacZip-Qwen3-8b

```bash
bash experiments/eval_token_level_selective_compression_taczip_qwen3_8b.sh
```

#### TacZip-Llama2-7b

```bash
bash experiments/eval_token_level_selective_compression_taczip_llama2_7b.sh
```

