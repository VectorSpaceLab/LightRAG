<p align="center">
  <img src="https://raw.githubusercontent.com/VectorSpaceLab/LightRAG/refs/heads/main/TacZip/assets/taczip_logo.png" width="65%">
</p>

<div align="center">
<a href="https://dl.acm.org/doi/abs/10.1145/3701551.3703580" target="_blank"><img src="https://img.shields.io/badge/ACM%20DL-Paper-blue?logo=acm"></a>
<a href="https://huggingface.co/wcyno23/TacZip-Qwen3-8b" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20TacZip--Qwen3--8b-orange"></a>
<a href="https://huggingface.co/wcyno23/TacZip-Llama2-7b" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20TacZip--Llama2--7b-orange"></a>
<a href="https://huggingface.co/datasets/wcyno23/TacZip-Data" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20TacZip--Data-ff69b4.svg"></a>
</div>
<h4 align="center">

### Environment

You can install the necessary dependencies using the following command. Recommended Python version is 3.10+.

```bash
conda create -n taczip python=3.10
conda activate taczip
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

### Usage

#### Inference

See [inference section](./examples/inference.md).

#### Evaluation

See [evaluation section](./example/evaluation.md).

#### Training

See [compressive encoder training section](./examples/training_compressive_encoder.md) and [token embedding training section](./examples/training_token_embedding.md).
