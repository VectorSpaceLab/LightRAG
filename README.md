<h1 align="center">LightRAG</h1>

## 1. Overview

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

- **FlexRAG** — Provides flexible context adaptation for question‑answering tasks.
- **TacZip** — Delivers task-aware context compression with fine-grained, token-level ratio allocation.

