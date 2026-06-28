---
title: "Understanding Transformer Attention: From Dot Products to Self-Attention"
date: 2026-06-28 10:00:00 +0000
categories: [Machine Learning]
tags: [transformers, attention, deep-learning, nlp]
math: true
---

The attention mechanism is the core building block of the Transformer architecture. This post builds it up from first principles so that the math and code land at the same time.

## The Problem Attention Solves

In a sequence model, we want each token to "look at" other tokens and decide how much weight to give each one when producing its output. Early RNN approaches processed sequences step-by-step, which made long-range dependencies hard to learn and slow to train.

Attention replaces sequential processing with a differentiable lookup: given a **query**, find the most relevant **keys** in the sequence and return a weighted sum of **values**.

## Scaled Dot-Product Attention

Given matrices $Q$ (queries), $K$ (keys), and $V$ (values), attention is:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

The $\sqrt{d_k}$ scaling prevents the dot products from growing too large in magnitude as dimension increases, which would push softmax into saturation.

## Implementation in Python

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / d_k ** 0.5

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, V), weights
```

## Multi-Head Attention

Instead of computing a single attention function, we project $Q$, $K$, $V$ into $h$ different subspaces, attend in each one independently, then concatenate and project back:

$$
\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\, W^O
$$

where each head $i$ is $\text{Attention}(QW_i^Q,\, KW_i^K,\, VW_i^V)$.

This lets the model jointly attend to information from different representation subspaces — some heads capture local dependencies, others long-range ones.

```python
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def forward(self, Q, K, V, mask=None):
        B = Q.size(0)
        Q = self.split_heads(self.W_q(Q), B)
        K = self.split_heads(self.W_k(K), B)
        V = self.split_heads(self.W_v(V), B)

        out, _ = scaled_dot_product_attention(Q, K, V, mask)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.num_heads * self.d_k)
        return self.W_o(out)
```

## Key Takeaways

| Property | Description |
|---|---|
| **Parallelism** | All positions attend simultaneously — no sequential bottleneck |
| **Dynamic weights** | Attention weights are input-dependent, not fixed |
| **Interpretability** | Attention maps can be visualised and sometimes interpreted |
| **Quadratic cost** | $O(n^2 d)$ in sequence length — the main scaling challenge |

The quadratic cost in sequence length is why efficient attention variants (Sparse Attention, Flash Attention, Linear Attention) have been active research areas. But for most practical sequence lengths, the standard implementation above is exactly what you need.

---

In the next post I'll cover positional encodings and why they matter for Transformers that have no inherent notion of order.
