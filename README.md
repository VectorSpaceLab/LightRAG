<h1 align="center">LightRAG</h1>

<div align="center">
<a href="https://dl.acm.org/doi/abs/10.1145/3701551.3703580" target="_blank"><img src="https://img.shields.io/badge/ACM%20DL-Paper-blue?logo=acm"></a>
<a href="https://huggingface.co/wcyno23/TacZip-Qwen3-8b" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20TacZip--Qwen3--8b-orange"></a>
<a href="https://huggingface.co/wcyno23/TacZip-Llama2-7b" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20TacZip--Llama2--7b-orange"></a>
<a href="https://huggingface.co/datasets/wcyno23/TacZip-Data" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20TacZip--Data-ff69b4.svg"></a>
<a href="https://huggingface.co/wcyno23/FlexRAG" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20FlexRAG%20Model-27b3b4.svg"></a>
<a href="https://github.com/"><img alt="License" src="https://img.shields.io/badge/Apache-2.0-green"></a>
</div>
<h4 align="center">

## 1. News

* 2026-02-03: ⚡Released task aware context compression model:  [TacZip-Qwen3-8b](https://huggingface.co/wcyno23/TacZip-Qwen3-8b) and [TacZip-Llama2-7b](https://huggingface.co/wcyno23/TacZip-Llama2-7b).
* 2026-01-21: We release the code for FlexRAG. Inference: [inference.md](https://github.com/VectorSpaceLab/LightRAG/tree/main/FlexRAG#inference) Evaluation: [evaluation.md](https://github.com/VectorSpaceLab/LightRAG/blob/main/FlexRAG/examples/evaluation.md) Train: [training.md](https://github.com/VectorSpaceLab/LightRAG/blob/main/FlexRAG/examples/training.md) Paper: [FlexRAG](https://dl.acm.org/doi/abs/10.1145/3701551.3703580).

## 2. Overview

**LightRAG** is a lightweight and efficient retrieval-augmented generation (RAG) framework that reduces compute overhead while maintaining strong generation quality. Instead of storing and attending to full embeddings of large contexts, it applies **latent context compression**, enabling scalable and efficient generation. The context is first converted into a **compressive embedding** and then **down-sampled** based on a target compression ratio. This ratio can be flexibly allocated in various ways, e.g., according to priority.

LightRAG is built around four core design principles:

- **Flexible compression ratios**
   Supports arbitrary compression ratios, allowing users to trade off efficiency and accuracy based on task and resource constraints.
- **Selective compression**
   Allocates compression budgets selectively, preserving semantically important information while keeping only a minimal amount of auxiliary context.
- **Unified multi-task compression**
   Compresses contexts from different tasks into a shared latent space, enabling efficient handling of heterogeneous and multi-task data.
- **Efficiency–quality balance**
   Explicitly balances computational efficiency and generation quality, ensuring performance remains stable even under aggressive compression.

The project includes two solutions:

* **TacZip** — Delivers task-aware context compression with fine-grained, token-level ratio allocation.

- **FlexRAG** — Provides flexible context adaptation for question‑answering tasks.

