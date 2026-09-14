---
title: "The KV Cache by Hand: A Complete Mathematical Walkthrough"
date: 2026-09-13 10:00:00 +0000
categories: [Machine Learning]
tags: [transformers, attention, kv-cache, llm, inference, gqa, paged-attention, quantization]
description: "Every computation behind the KV cache carried out in vector and matrix form with actual numbers: what the cache holds, why reusing it is exact, the work it saves, and what it costs in memory and bandwidth."
math: true
---

The KV cache is usually described in one sentence: store the keys and values so they are not recomputed. This post carries out every computation behind that sentence, in vector and matrix form with the actual numbers: what the cache holds, why reusing it is exact, what it saves, and what it costs in memory and bandwidth.

It is the mathematical companion to [Part 2 of *From Attention to Prompt Caching*](https://chuan-zhang.medium.com/from-attention-to-prompt-caching-a-trilogy-2-79934b836de5), and it continues [the Part 1 walkthrough]({% post_url 2026-09-12-attention-by-hand-mathematical-walkthrough %}). The toy-model numbers are reproduced from [`toy_llm_demo.py`](https://gist.github.com/chuan2019/c9b632faeb9422ca2680af9ca6066dad) and the two-layer numbers from [`kv_cache_demo.py`](https://gist.github.com/chuan2019/f2bf188647210f39a7c987dca6be62bf), both dependency-free Python.

---

## 1. Setup

### 1.1 The toy model, recalled

The toy model of Part 1 has the vocabulary

$$
\begin{array}{c|ccccccccc}
\text{token} & \texttt{<bos>} & \texttt{cat} & \texttt{->} & \texttt{meow} & \texttt{;} & \texttt{dog} & \texttt{woof} & \texttt{cow} & \texttt{moo}\\ \hline
\text{index} & 0 & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8
\end{array}
$$

one-hot embeddings $$e(x) = \mathbf{e}_{\mathrm{id}(x)} \in \mathbb{R}^9$$, and a single attention head with the hand-set rules

$$
k_i = e(x_{i-2}) + e(x_{i-1}), \qquad v_i = e(x_i), \qquad q_t = e(x_{t-1}) + e(x_t), \qquad s_i = \lambda\,(q_t \cdot k_i),\ \lambda = 4,
$$

where $$x_i := \texttt{<bos>}$$ for $$i < 0$$. The prompt $$\texttt{cat -> meow ; dog -> woof ; cow -> moo ; dog ->}$$ has $$n_0 = 14$$ tokens, and generation is greedy for $$T = 6$$ steps. Positions are numbered from $$0$$, so step $$s$$ predicts the token after position $$t = 13 + s$$.

### 1.2 The cache as two growing matrices

Write $$m$$ for the number of positions whose key and value have been computed. The cache is the pair

$$
K^{(m)} = \begin{bmatrix} k_0^\top \\ k_1^\top \\ \vdots \\ k_{m-1}^\top \end{bmatrix} \in \mathbb{R}^{m \times 9},
\qquad
V^{(m)} = \begin{bmatrix} v_0^\top \\ v_1^\top \\ \vdots \\ v_{m-1}^\top \end{bmatrix} \in \mathbb{R}^{m \times 9}.
$$

Prefill sets $$m = n_0 = 14$$. Each decode step reads the whole cache and then appends exactly one row to each matrix:

$$
K^{(m+1)} = \begin{bmatrix} K^{(m)} \\ k_m^\top \end{bmatrix},
\qquad
V^{(m+1)} = \begin{bmatrix} V^{(m)} \\ v_m^\top \end{bmatrix}.
$$

No row already in $$K^{(m)}$$ or $$V^{(m)}$$ is ever modified, and §3.2 proves that none needs to be.

### 1.3 The two-layer demo

A one-layer model cannot exhibit the depth term of the post, so §4.3, §6.3 and §6.4 also use `kv_cache_demo.py`: $$L = 2$$ attention layers with a residual stream, width $$d = 8$$, a 16-token vocabulary, fixed pseudo-random weights, a 20-token prompt and $$T = 10$$ generated tokens. It uses the row-vector convention, so a projection is $$h W$$ with $$W \in \mathbb{R}^{8 \times 8}$$, and each layer computes

$$
q = h W_Q, \quad k = h W_K, \quad v = h W_V, \qquad
w = \mathrm{softmax}\!\left(\frac{K\, q^\top}{\sqrt{8}}\right), \qquad
h' = h + (w V)\, W_O,
$$

where $$K$$ and $$V$$ are that layer's cache after this token's $$k$$ and $$v$$ have been appended, so each token attends to itself and to every earlier position.

---

## 2. The redundancy the cache removes

### 2.1 One step, two ways

To predict the token after position $$t$$, a run with a cache and a run without one both evaluate

$$
\mathrm{out}_t = \mathrm{softmax}\!\big(\lambda\, q_t^\top K^\top\big)\, V, \qquad K, V \in \mathbb{R}^{(t+1) \times 9}.
$$

They differ only in where $$K$$ and $$V$$ come from. The cacheless run rebuilds all $$t + 1$$ rows from the tokens. The cached run already holds rows $$0, \dots, t$$, because it projected row $$t$$ when token $$x_t$$ was produced, and computes nothing new before reading them. Both apply the same rule to the same tokens, so the matrices are equal row for row, and so are the outputs.

### 2.2 Counting the work in the toy

Counting one key–value projection per row computed,

$$
\text{no cache: } \sum_{s=0}^{T-1} (n_0 + s) = T n_0 + \frac{T(T-1)}{2} = 6 \cdot 14 + 15 = 99,
\qquad
\text{cache: } n_0 + T = 14 + 6 = 20,
$$

a ratio of $$99/20 = 4.95$$. The general ratio

$$
\frac{T n_0 + T(T-1)/2}{n_0 + T}
$$

tends to $$T$$ as $$n_0 \to \infty$$ and to $$T/2$$ as $$T \to \infty$$, so it grows without bound with the length of the generation.

One detail of the count: the cached run appends the row of the last generated token, $$(k_{19}, v_{19})$$, after producing it, and no later step reads that row. A cache that skipped it would make $$19$$ projections, a ratio of $$99/19 = 5.211$$. The script counts the row it actually writes.

Score operations, one per query–key dot product, come out equal:

$$
\text{no cache: } \sum_{s=0}^{5} (14 + s) = 99, \qquad \text{cache: } \sum_{s=0}^{5} (14 + s) = 99.
$$

The toy has one layer and reads only the last position's output, so even the cacheless run needs a single query row per step. §6.3 shows what a second layer changes.

### 2.3 In FLOPs

A forward pass costs about $$2P$$ FLOPs per token processed, for $$P$$ parameters. At step $$s$$ the cacheless run processes $$n_0 + s$$ tokens and the cached run processes one:

$$
\text{no cache: } 2P\,(n_0 + s), \qquad \text{cache: } 2P.
$$

Over the whole generation that is $$2P\big(T n_0 + T(T-1)/2\big)$$ against $$2P\,(n_0 + T)$$: the 99-to-20 shape of §2.2, scaled by $$2P$$.

### 2.4 The attention grid

The post's Figure 1 shows step 6 of a 7-token sequence. A causal sequence of length $$m$$ has $$m(m+1)/2$$ query–key cells, and a cached step computes only the last row of them:

$$
m = 7: \qquad \text{no cache: } \frac{7 \cdot 8}{2} = 28 \text{ cells}, \qquad \text{cache: } 7 \text{ cells}.
$$

The one-layer toy needs only that last row even without a cache. §6.3 explains why a deeper model needs the whole triangle.

---

## 3. Why K and V, and not Q

### 3.1 Who reads each vector

In the cached run, step $$s$$ reads the $$m = 14 + s$$ rows $$0, \dots, 13 + s$$, and uses its query once. Row $$i$$ is read at every step with $$14 + s > i$$:

$$
r_i = \#\{\, s \in \{0, \dots, 5\} : 14 + s > i \,\} =
\begin{cases}
6 & 0 \le i \le 13,\\
19 - i & 14 \le i \le 19.
\end{cases}
$$

$$
\begin{array}{c|ccccccc}
i & 0\text{–}13 & 14 & 15 & 16 & 17 & 18 & 19\\ \hline
\text{reads of } k_i \text{ and } v_i & 6 & 5 & 4 & 3 & 2 & 1 & 0
\end{array}
$$

Each of the six queries $$q_{13}, \dots, q_{18}$$ is read exactly once. The total, $$6 \cdot 14 + 5 + 4 + 3 + 2 + 1 = 99$$, is the score-operation count of §2.2, since every score reads one stored key. Storing a query would store something with no second reader. Storing a key or a value turns each of its later reads into a lookup.

### 3.2 The invariant, proved

Let $$h_i^{(\ell)}$$ be the vector at position $$i$$ entering layer $$\ell$$, with $$h_i^{(0)}$$ the embedding of $$x_i$$ plus, in a real model, an encoding of position $$i$$. A causal layer computes

$$
h_i^{(\ell+1)} = F^{(\ell)}\Big(h_i^{(\ell)},\ \big\{ \big(k_j^{(\ell)}, v_j^{(\ell)}\big) : j \le i \big\}\Big),
\qquad
k_j^{(\ell)} = h_j^{(\ell)} W_K^{(\ell)}, \quad v_j^{(\ell)} = h_j^{(\ell)} W_V^{(\ell)},
$$

where $$F^{(\ell)}$$ collects the query at $$i$$, the masked softmax, the output projection, the residual additions and the feed-forward block. Apart from the attention read, all of these act on position $$i$$ alone.

**Claim.** For every layer $$\ell$$ and position $$i$$, the vectors $$h_i^{(\ell)}$$, $$k_i^{(\ell)}$$ and $$v_i^{(\ell)}$$ depend only on $$x_0, \dots, x_i$$.

**Proof,** by induction on $$\ell$$. For $$\ell = 0$$, $$h_i^{(0)}$$ depends only on $$x_i$$ and the fixed position $$i$$. Suppose the claim holds at layer $$\ell$$. For $$j \le i$$, $$k_j^{(\ell)}$$ and $$v_j^{(\ell)}$$ depend only on $$x_0, \dots, x_j$$, which lie among $$x_0, \dots, x_i$$, and so does $$h_i^{(\ell)}$$. The weights are fixed at inference, so $$h_i^{(\ell+1)}$$ is a fixed function of these and depends only on $$x_0, \dots, x_i$$, and so are $$k_i^{(\ell+1)}$$ and $$v_i^{(\ell+1)}$$, which are fixed linear maps of it. $$\blacksquare$$

Appending $$x_{i+1}, x_{i+2}, \dots$$ therefore cannot change any stored $$k_i^{(\ell)}$$ or $$v_i^{(\ell)}$$, and every row of every layer's cache stays valid for the rest of the request. The proof used exactly the three properties the post lists: the mask ($$j \le i$$), frozen weights, and a query that is read only at its own position. The post indexes tokens from $$1$$; the statement is the same.

### 3.3 Exactness in numbers

Both scripts compare the two runs directly:

$$
\max_{\text{steps},\ j} \big\lvert p^{\text{cache}}_j - p^{\text{no cache}}_j \big\rvert = 0 \quad \text{(toy, 6 steps)},
\qquad
\max_{\text{steps},\ j} \big\lvert \ell^{\text{cache}}_j - \ell^{\text{no cache}}_j \big\rvert = 0 \quad \text{(two-layer demo, 10 steps)}.
$$

The generated sequences also match token for token: $$\texttt{woof ; cow -> moo ;}$$ in the toy and $$[8, 5, 8, 5, 8, 5, 8, 5, 8, 5]$$ in the demo.

---

## 4. Testing the invariant by hand

### 4.1 The toy probe

Probe $$k_6$$ under three sequences. In the baseline,

$$
k_6 = e(x_4) + e(x_5) = e(\texttt{dog}) + e(\texttt{->}) = \mathbf{e}_5 + \mathbf{e}_2 = \begin{bmatrix}0&0&1&0&0&1&0&0&0\end{bmatrix}^\top.
$$

If everything after position 6 is replaced, by $$\texttt{moo ; cat}$$ in the script, then $$x_4$$ and $$x_5$$ are untouched and

$$
k_6' = \mathbf{e}_5 + \mathbf{e}_2, \qquad \big\lVert k_6' - k_6 \big\rVert_\infty = 0.
$$

If instead the token at position 5 is changed from $$\texttt{->}$$ to $$\texttt{cow}$$,

$$
k_6'' = e(\texttt{dog}) + e(\texttt{cow}) = \mathbf{e}_5 + \mathbf{e}_7,
\qquad
k_6'' - k_6 = \mathbf{e}_7 - \mathbf{e}_2 = \begin{bmatrix}0&0&-1&0&0&0&0&1&0\end{bmatrix}^\top,
$$

$$
\big\lVert k_6'' - k_6 \big\rVert_\infty = 1, \qquad \big\lVert k_6'' - k_6 \big\rVert_2 = \sqrt{2}.
$$

The changed row no longer answers the lookup. The step-0 query $$q_{13} = \mathbf{e}_2 + \mathbf{e}_5$$ scores $$q_{13} \cdot k_6'' = 1$$ against it instead of $$2$$, and the step-0 prediction becomes $$\texttt{;}$$ with $$p = 0.2831$$, while $$p(\texttt{woof}) = 0.1403$$.

### 4.2 Why the toy probe proves less

The key $$k_6$$ reads only $$x_4$$ and $$x_5$$, so changing $$x_1$$ would also give $$\lVert \Delta k_6 \rVert_\infty = 0$$, not because of causality but because of the two-token window. The half of the invariant that the cache relies on, that no key depends on a later token, holds either way.

### 4.3 The two-layer probe

The demo repeats the test on the layer-1 key at position 2, $$k_2^{(1)} = h_2^{(1)} W_K^{(1)}$$:

$$
\begin{array}{l|c}
\text{sequence} & \big\lVert \Delta k_2^{(1)} \big\rVert_\infty\\ \hline
\text{same first three tokens, different continuation} & 0\\
\text{token at position 1 changed} & 0.132
\end{array}
$$

This time a change at position 1 does register, even though position 1 is not part of any window. The layer-0 key at position 2 depends on $$x_2$$ alone and is unchanged, $$\lVert \Delta k_2^{(0)} \rVert_\infty = 0$$. But layer 0 at position 2 attends to positions 0, 1 and 2 and adds the result to $$h_2^{(1)}$$, so the layer-1 key has absorbed $$x_1$$ through the residual stream. With more layers, $$k_i$$ comes to depend on the whole prefix, which §3.2 allows, and never on anything after it, which §3.2 forbids.

---

## 5. What is in the cache

### 5.1 The full cache after generation

After six generated tokens the cached run has written $$m = 20$$ rows. Each line below gives $$i$$, the nine entries of $$k_i$$, the nine entries of $$v_i$$, and $$x_i$$, with columns ordered $$(\texttt{<bos>},\texttt{cat},\texttt{->},\texttt{meow},\texttt{;},\texttt{dog},\texttt{woof},\texttt{cow},\texttt{moo})$$. Rows $$0$$–$$13$$ come from prefill, and rows $$14$$–$$19$$, below the rule, from decode.

$$
\begin{array}{r|ccccccccc|ccccccccc|l}
0 & 2&0&0&0&0&0&0&0&0 & 0&1&0&0&0&0&0&0&0 & \texttt{cat} \\
1 & 1&1&0&0&0&0&0&0&0 & 0&0&1&0&0&0&0&0&0 & \texttt{->} \\
2 & 0&1&1&0&0&0&0&0&0 & 0&0&0&1&0&0&0&0&0 & \texttt{meow} \\
3 & 0&0&1&1&0&0&0&0&0 & 0&0&0&0&1&0&0&0&0 & \texttt{;} \\
4 & 0&0&0&1&1&0&0&0&0 & 0&0&0&0&0&1&0&0&0 & \texttt{dog} \\
5 & 0&0&0&0&1&1&0&0&0 & 0&0&1&0&0&0&0&0&0 & \texttt{->} \\
6 & 0&0&1&0&0&1&0&0&0 & 0&0&0&0&0&0&1&0&0 & \texttt{woof} \\
7 & 0&0&1&0&0&0&1&0&0 & 0&0&0&0&1&0&0&0&0 & \texttt{;} \\
8 & 0&0&0&0&1&0&1&0&0 & 0&0&0&0&0&0&0&1&0 & \texttt{cow} \\
9 & 0&0&0&0&1&0&0&1&0 & 0&0&1&0&0&0&0&0&0 & \texttt{->} \\
10 & 0&0&1&0&0&0&0&1&0 & 0&0&0&0&0&0&0&0&1 & \texttt{moo} \\
11 & 0&0&1&0&0&0&0&0&1 & 0&0&0&0&1&0&0&0&0 & \texttt{;} \\
12 & 0&0&0&0&1&0&0&0&1 & 0&0&0&0&0&1&0&0&0 & \texttt{dog} \\
13 & 0&0&0&0&1&1&0&0&0 & 0&0&1&0&0&0&0&0&0 & \texttt{->} \\
\hline 14 & 0&0&1&0&0&1&0&0&0 & 0&0&0&0&0&0&1&0&0 & \texttt{woof} \\
15 & 0&0&1&0&0&0&1&0&0 & 0&0&0&0&1&0&0&0&0 & \texttt{;} \\
16 & 0&0&0&0&1&0&1&0&0 & 0&0&0&0&0&0&0&1&0 & \texttt{cow} \\
17 & 0&0&0&0&1&0&0&1&0 & 0&0&1&0&0&0&0&0&0 & \texttt{->} \\
18 & 0&0&1&0&0&0&0&1&0 & 0&0&0&0&0&0&0&0&1 & \texttt{moo} \\
19 & 0&0&1&0&0&0&0&0&1 & 0&0&0&0&1&0&0&0&0 & \texttt{;}
\end{array}
$$

The middle block is $$K^{(20)} \in \mathbb{R}^{20 \times 9}$$ and the right block is $$V^{(20)} \in \mathbb{R}^{20 \times 9}$$.

### 5.2 Row 6 holds the answer

At step 0 the query is $$q_{13} = e(\texttt{dog}) + e(\texttt{->}) = \mathbf{e}_2 + \mathbf{e}_5$$. Against the prompt rows of the cache,

$$
K^{(14)} q_{13} = \begin{bmatrix}0&0&1&1&0&1&2&1&0&0&1&1&0&1\end{bmatrix}^\top,
$$

and row 6 is the only row with two matches. Its value $$V_6 = \mathbf{e}_6 = e(\texttt{woof})$$ enters the output with weight $$0.884781734$$. The lookup reads the answer directly out of a stored row.

### 5.3 Prefill rows and decode rows are indistinguishable

Rows $$0$$–$$13$$ were written in one pass and rows $$14$$–$$19$$ one per step, but each is the same function of the tokens before it, so the matrix records nothing about which is which. If $$K_{\text{prefill}} \in \mathbb{R}^{14 \times 9}$$ is the key matrix computed in one pass over the prompt, then

$$
K^{(20)}_{0:14} = K_{\text{prefill}},
$$

and rows $$14$$ to $$19$$ are exactly the rows that one-pass prefill over the 20-token sequence would have produced.

### 5.4 Repeated context, repeated rows

The toy has no positional encoding, so identical context gives identical rows. Row 13 repeats row 5, and because the generated text $$\texttt{woof ; cow -> moo ;}$$ repeats positions 6–11 of the prompt, every decode row repeats an earlier one:

$$
K_{13} = K_5, \quad V_{13} = V_5, \qquad K_{14+j} = K_{6+j}, \quad V_{14+j} = V_{6+j} \qquad (j = 0, \dots, 5).
$$

$$K^{(20)}$$ therefore has only $$13$$ distinct rows out of 20. A real model applies RoPE, which rotates pairs of coordinates of $$q_i$$ and $$k_i$$ by angles proportional to $$i$$, so none of these rows would coincide. That position dependence is the constraint Part 3 works within.

### 5.5 Size

The toy cache holds $$20 \times 2 \times 9 = 360$$ numbers for one head in one layer. The two-layer demo ends holding $$L \times 2 \times m \times d = 2 \times 2 \times 30 \times 8 = 960$$. §8 attaches units to the same product.

---

## 6. What the cache saves

### 6.1 Decode, one step at a time

Step 0 reads $$K^{(14)}$$ and $$V^{(14)}$$. With the dot products of §5.2, the scores take the values $$8$$, $$4$$ and $$0$$ on $$n_2 = 1$$, $$n_1 = 7$$ and $$n_0 = 6$$ rows, so the softmax normalizer, taken relative to the largest score, is

$$
Z = 1 + 7\,e^{-4} + 6\,e^{-8} = 1 + 7\,(0.018315639) + 6\,(0.000335463) = 1.13022225,
$$

and the top weight is $$w_6 = 1/Z = 0.884781734 = p(\texttt{woof})$$. The step then writes one row for the new position 14, whose two preceding tokens are $$x_{12} = \texttt{dog}$$ and $$x_{13} = \texttt{->}$$:

$$
k_{14} = e(\texttt{dog}) + e(\texttt{->}) = \begin{bmatrix}0&0&1&0&0&1&0&0&0\end{bmatrix}^\top, \qquad v_{14} = e(\texttt{woof}) = \begin{bmatrix}0&0&0&0&0&0&1&0&0\end{bmatrix}^\top,
$$

$$
K^{(15)} = \begin{bmatrix} K^{(14)} \\ k_{14}^\top \end{bmatrix} \in \mathbb{R}^{15 \times 9}, \qquad V^{(15)} = \begin{bmatrix} V^{(14)} \\ v_{14}^\top \end{bmatrix} \in \mathbb{R}^{15 \times 9}.
$$

Step 1 reads these with $$q_{14} = e(\texttt{->}) + e(\texttt{woof})$$. All six steps, where $$n_1$$ and $$n_0$$ count the rows scoring $$4$$ and $$0$$ (one row always scores $$8$$):

$$
\begin{array}{c|c|c|l|c|c|l|l|l}
s & t & m & q_t & n_1 & n_0 & Z & \text{top token }(p) & \text{runner-up }(p) \\ \hline
0 & 13 & 14 & e(\texttt{dog}) + e(\texttt{->}) & 7 & 6 & 1.13022225 & \texttt{woof}\ (0.884782) & \texttt{;}\ (0.048616) \\
1 & 14 & 15 & e(\texttt{->}) + e(\texttt{woof}) & 7 & 7 & 1.13055771 & \texttt{;}\ (0.916920) & \texttt{woof}\ (0.032401) \\
2 & 15 & 16 & e(\texttt{woof}) + e(\texttt{;}) & 7 & 8 & 1.13089317 & \texttt{cow}\ (0.884257) & \texttt{->}\ (0.048884) \\
3 & 16 & 17 & e(\texttt{;}) + e(\texttt{cow}) & 7 & 9 & 1.13122864 & \texttt{->}\ (0.916673) & \texttt{dog}\ (0.032382) \\
4 & 17 & 18 & e(\texttt{cow}) + e(\texttt{->}) & 9 & 8 & 1.16752445 & \texttt{moo}\ (0.856513) & \texttt{;}\ (0.062750) \\
5 & 18 & 19 & e(\texttt{->}) + e(\texttt{moo}) & 9 & 9 & 1.16785991 & \texttt{;}\ (0.903316) & \texttt{woof}\ (0.031366)
\end{array}
$$

Every step appends one row and rewrites none. The cache grows from 14 to 19 rows read while the projection work per step stays at one row. The slow fall in the top probability is the growth of $$Z$$ with $$n_1$$ and $$n_0$$, explained in the Part 1 walkthrough, §8.4.

### 6.2 Identical to recomputing

Step $$s$$ of the cacheless run rebuilds $$K$$ and $$V$$ with $$14 + s$$ rows by applying the same rule to the same tokens, so it produces exactly $$K^{(14+s)}$$ and $$V^{(14+s)}$$ and the same output. The difference is that it makes $$14 + 15 + \dots + 19 = 99$$ projections to reach the matrices the cached run holds after $$20$$.

### 6.3 Depth turns the saving into a requirement

With $$L$$ layers, the input to layer $$\ell + 1$$ at position $$i$$ is layer $$\ell$$'s output at position $$i$$. A run without a cache must therefore compute attention at every position of every layer at every step. Step $$s$$ processes a sequence of length $$n_0 + s$$, in which position $$p$$ attends to $$p + 1$$ keys:

$$
\text{score ops (no cache)} = L \sum_{s=0}^{T-1} \sum_{p=0}^{n_0+s-1} (p+1) = L \sum_{s=0}^{T-1} \frac{(n_0+s)(n_0+s+1)}{2},
$$

$$
\text{score ops (cache)} = L \sum_{p=0}^{n_0+T-1} (p+1) = L\,\frac{(n_0+T)(n_0+T+1)}{2}.
$$

The first is a sum of quadratics, $$\Theta(T^3)$$ when $$T \gg n_0$$; the second is $$\Theta(T^2)$$. For the demo, with $$L = 2$$, $$n_0 = 20$$ and $$T = 10$$,

$$
\text{no cache: } 2\,(210 + 231 + 253 + 276 + 300 + 325 + 351 + 378 + 406 + 435) = 2 \cdot 3165 = 6330,
$$

$$
\text{cache: } 2 \cdot \frac{30 \cdot 31}{2} = 930, \qquad \frac{6330}{930} = 6.806,
$$

and for projections, $$2 \sum_{s=0}^{9} (20 + s) = 490$$ against $$2 \cdot 30 = 60$$, a ratio of $$8.167$$. These are exactly the counts the script reports, $$6.8\times$$ and $$8.2\times$$.

With a single layer only the last position of each step would need attention, and both runs would do $$\sum_{s=0}^{9} (20 + s) = 245$$ score operations: the flat pair of bars in the post's Figure 4. The second layer is what makes recomputation cubic.

Like the toy, the demo's cached run also processes the tenth generated token, whose logits are never used. Without that final pass the cache would make $$870$$ score operations and $$58$$ projections, ratios of $$7.276$$ and $$8.448$$.

### 6.4 One position through both layers

To see what a deeper cache holds, follow position 2 of the demo prompt, token $$7$$, through both layers during prefill. Positions 0 and 1 have already written their rows.

**Layer 0.** The input is row $$x_2$$ of the embedding matrix:

$$
h_2^{(0)} = \begin{bmatrix}-0.4149 & -0.1938 & 0.2640 & -0.4321 & 0.3560 & -0.1200 & 0.4168 & -0.0331\end{bmatrix}
$$

$$
\begin{aligned}
q_2^{(0)} &= h_2^{(0)} W_Q^{(0)} = \begin{bmatrix}0.2640 & -0.3506 & -0.2897 & 0.0260 & 0.1845 & 0.2941 & 0.4232 & 0.1481\end{bmatrix}\\
k_2^{(0)} &= h_2^{(0)} W_K^{(0)} = \begin{bmatrix}-0.2036 & 0.4968 & -0.1343 & 0.1414 & 0.0418 & 0.0773 & -0.2920 & -0.0670\end{bmatrix}\\
v_2^{(0)} &= h_2^{(0)} W_V^{(0)} = \begin{bmatrix}-0.0921 & 0.0728 & 0.3484 & 0.3838 & -0.1093 & 0.2728 & 0.1923 & 0.0757\end{bmatrix}
\end{aligned}
$$

After the append, layer 0's cache holds three rows:

$$
K^{(0)}_{0:3} = \begin{bmatrix}0.3085 & -0.1387 & 0.1898 & -0.3440 & 0.2347 & 0.2163 & 0.0271 & 0.1929 \\ 0.2919 & -0.1023 & 0.0631 & -0.4698 & 0.0920 & 0.0092 & 0.2528 & 0.0919 \\ -0.2036 & 0.4968 & -0.1343 & 0.1414 & 0.0418 & 0.0773 & -0.2920 & -0.0670\end{bmatrix}
$$

$$
V^{(0)}_{0:3} = \begin{bmatrix}0.1733 & -0.1779 & 0.1341 & 0.0976 & -0.1991 & -0.1465 & 0.0568 & -0.1913 \\ 0.1254 & 0.0916 & -0.0707 & 0.1217 & -0.4441 & -0.0565 & 0.4548 & -0.2268 \\ -0.0921 & 0.0728 & 0.3484 & 0.3838 & -0.1093 & 0.2728 & 0.1923 & 0.0757\end{bmatrix}
$$

$$
s = \frac{K^{(0)}_{0:3}\, q_2^{(0)\top}}{\sqrt{8}} = \begin{bmatrix}0.0754 & 0.0787 & -0.1020\end{bmatrix}^\top, \qquad w = \mathrm{softmax}(s) = \begin{bmatrix}0.3520 & 0.3532 & 0.2948\end{bmatrix}^\top
$$

$$
w\,V^{(0)}_{0:3} = \begin{bmatrix}0.0781 & -0.0088 & 0.1249 & 0.1905 & -0.2592 & 0.0089 & 0.2373 & -0.1251\end{bmatrix}
$$

$$
h_2^{(1)} = h_2^{(0)} + \big(w\,V^{(0)}_{0:3}\big) W_O^{(0)} = h_2^{(0)} + \begin{bmatrix}0.0766 & 0.1246 & 0.0329 & 0.0774 & -0.1216 & 0.1848 & -0.1268 & 0.1062\end{bmatrix} = \begin{bmatrix}-0.3383 & -0.0693 & 0.2969 & -0.3547 & 0.2345 & 0.0648 & 0.2900 & 0.0732\end{bmatrix}
$$

**Layer 1.** The same computation on $$h_2^{(1)}$$:

$$
\begin{aligned}
q_2^{(1)} &= h_2^{(1)} W_Q^{(1)} = \begin{bmatrix}-0.0881 & -0.2032 & 0.0685 & 0.1262 & 0.1173 & -0.1038 & -0.4271 & -0.1621\end{bmatrix}\\
k_2^{(1)} &= h_2^{(1)} W_K^{(1)} = \begin{bmatrix}0.3848 & 0.0690 & -0.2583 & 0.0087 & -0.1501 & -0.4039 & -0.1339 & -0.0287\end{bmatrix}\\
v_2^{(1)} &= h_2^{(1)} W_V^{(1)} = \begin{bmatrix}-0.1020 & 0.0577 & -0.4142 & -0.1466 & -0.0517 & -0.0976 & 0.3645 & 0.0739\end{bmatrix}
\end{aligned}
$$

$$
K^{(1)}_{0:3} = \begin{bmatrix}-0.2353 & -0.1190 & 0.0506 & 0.0470 & 0.2013 & 0.1196 & 0.2045 & 0.3261 \\ -0.4069 & -0.1031 & -0.1242 & -0.0347 & -0.0467 & -0.0039 & 0.2571 & 0.1472 \\ 0.3848 & 0.0690 & -0.2583 & 0.0087 & -0.1501 & -0.4039 & -0.1339 & -0.0287\end{bmatrix}
$$

$$
V^{(1)}_{0:3} = \begin{bmatrix}0.2727 & -0.2869 & 0.0100 & 0.4130 & 0.0638 & 0.1180 & 0.0291 & -0.3397 \\ 0.1869 & -0.0487 & 0.1716 & 0.3446 & -0.1382 & 0.0519 & -0.2356 & -0.2376 \\ -0.1020 & 0.0577 & -0.4142 & -0.1466 & -0.0517 & -0.0976 & 0.3645 & 0.0739\end{bmatrix}
$$

$$
s = \begin{bmatrix}-0.0264 & -0.0335 & 0.0077\end{bmatrix}^\top, \qquad w = \begin{bmatrix}0.3303 & 0.3280 & 0.3418\end{bmatrix}^\top
$$

$$
h_2^{(2)} = h_2^{(1)} + \big(w\,V^{(1)}_{0:3}\big) W_O^{(1)} = h_2^{(1)} + \begin{bmatrix}0.1286 & 0.0310 & 0.0329 & -0.0894 & -0.1280 & -0.0438 & -0.0757 & 0.0035\end{bmatrix} = \begin{bmatrix}-0.2097 & -0.0383 & 0.3299 & -0.4441 & 0.1064 & 0.0210 & 0.2144 & 0.0767\end{bmatrix}
$$

This position writes two rows. The layer-0 pair $$(k_2^{(0)}, v_2^{(0)})$$ depends on $$x_2$$ alone. The layer-1 pair was computed from $$h_2^{(1)}$$, which already contains the weighted mix of positions 0, 1 and 2 above. A cacheless run that needs $$k_2^{(1)}$$ again must repeat the whole layer-0 computation at position 2 to rebuild $$h_2^{(1)}$$, which is the per-position, per-layer work counted in §6.3. The cache keeps the result.

---

## 7. The two shapes: prefill and decode

### 7.1 Prefill as one masked matrix product

Prefill can compute every prompt position's output at once. Stack the prompt's queries, $$q_i = e(x_{i-1}) + e(x_i)$$, as rows:

$$
Q = \begin{bmatrix}
1&1&0&0&0&0&0&0&0\\
0&1&1&0&0&0&0&0&0\\
0&0&1&1&0&0&0&0&0\\
0&0&0&1&1&0&0&0&0\\
0&0&0&0&1&1&0&0&0\\
0&0&1&0&0&1&0&0&0\\
0&0&1&0&0&0&1&0&0\\
0&0&0&0&1&0&1&0&0\\
0&0&0&0&1&0&0&1&0\\
0&0&1&0&0&0&0&1&0\\
0&0&1&0&0&0&0&0&1\\
0&0&0&0&1&0&0&0&1\\
0&0&0&0&1&1&0&0&0\\
0&0&1&0&0&1&0&0&0
\end{bmatrix} \in \mathbb{R}^{14 \times 9}
$$

The dot products of every query with every key form $$D = Q K^\top \in \mathbb{R}^{14 \times 14}$$. Entries above the diagonal are masked, and are shown as $$\cdot$$:

$$
D = \begin{array}{cccccccccccccc}
2 & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot \\
0 & 1 & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot \\
0 & 0 & 1 & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot \\
0 & 0 & 0 & 1 & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot \\
0 & 0 & 0 & 0 & 1 & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot \\
0 & 0 & 1 & 1 & 0 & 1 & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot \\
0 & 0 & 1 & 1 & 0 & 0 & 1 & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot \\
0 & 0 & 0 & 0 & 1 & 1 & 0 & 1 & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot \\
0 & 0 & 0 & 0 & 1 & 1 & 0 & 0 & 1 & \cdot & \cdot & \cdot & \cdot & \cdot \\
0 & 0 & 1 & 1 & 0 & 0 & 1 & 1 & 0 & 1 & \cdot & \cdot & \cdot & \cdot \\
0 & 0 & 1 & 1 & 0 & 0 & 1 & 1 & 0 & 0 & 1 & \cdot & \cdot & \cdot \\
0 & 0 & 0 & 0 & 1 & 1 & 0 & 0 & 1 & 1 & 0 & 1 & \cdot & \cdot \\
0 & 0 & 0 & 0 & 1 & 2 & 1 & 0 & 1 & 1 & 0 & 0 & 1 & \cdot \\
0 & 0 & 1 & 1 & 0 & 1 & 2 & 1 & 0 & 0 & 1 & 1 & 0 & 1
\end{array}
$$

$$
S = \lambda D + M, \qquad M_{ij} = \begin{cases} 0 & j \le i,\\ -\infty & j > i, \end{cases} \qquad W = \mathrm{softmax}_{\text{row}}(S), \qquad \mathrm{OUT} = W V \in \mathbb{R}^{14 \times 9}.
$$

The last row of $$\mathrm{OUT}$$ is the decode step-0 output, with $$\max_j \lvert \mathrm{OUT}_{13,j} - \mathrm{out}^{(0)}_j \rvert = 0$$. The other rows are each prompt position's own prediction:

$$
\begin{array}{r|l|c|l|c|l}
i & q_i & \max_{j \le i} D_{ij} & \text{prediction} & p & \text{actual } x_{i+1} \\ \hline
0 & e(\texttt{<bos>}) + e(\texttt{cat}) & 2 & \texttt{cat} & 1.000 & \texttt{->} \\
1 & e(\texttt{cat}) + e(\texttt{->}) & 1 & \texttt{->} & 0.982 & \texttt{meow} \\
2 & e(\texttt{->}) + e(\texttt{meow}) & 1 & \texttt{meow} & 0.965 & \texttt{;} \\
3 & e(\texttt{meow}) + e(\texttt{;}) & 1 & \texttt{;} & 0.948 & \texttt{dog} \\
4 & e(\texttt{;}) + e(\texttt{dog}) & 1 & \texttt{dog} & 0.932 & \texttt{->} \\
5 & e(\texttt{dog}) + e(\texttt{->}) & 1 & \texttt{->} & 0.333 & \texttt{woof} \\
6 & e(\texttt{->}) + e(\texttt{woof}) & 1 & \texttt{meow} & 0.325 & \texttt{;} \\
7 & e(\texttt{woof}) + e(\texttt{;}) & 1 & \texttt{->} & 0.329 & \texttt{cow} \\
8 & e(\texttt{;}) + e(\texttt{cow}) & 1 & \texttt{->} & 0.327 & \texttt{->} \\
9 & e(\texttt{cow}) + e(\texttt{->}) & 1 & \texttt{;} & 0.393 & \texttt{moo} \\
10 & e(\texttt{->}) + e(\texttt{moo}) & 1 & \texttt{;} & 0.391 & \texttt{;} \\
11 & e(\texttt{moo}) + e(\texttt{;}) & 1 & \texttt{->} & 0.394 & \texttt{dog} \\
12 & e(\texttt{;}) + e(\texttt{dog}) & 2 & \texttt{->} & 0.931 & \texttt{->} \\
13 & e(\texttt{dog}) + e(\texttt{->}) & 2 & \texttt{woof} & 0.885 & \texttt{woof}\ (\text{generated})
\end{array}
$$

Where the pair ending at position $$i$$ occurred earlier, as at rows 12 and 13, some key scores $$2$$ and the lookup finds the continuation. At row 5, the pair $$\texttt{dog ->}$$ has not occurred before, since its match lies in the future, so no key scores $$2$$ and the prediction is a near-uniform blend. Rows 1–4 have no earlier match and mostly copy their own token, and row 0's perfect match is an artifact of padding: $$q_0$$ and $$k_0$$ both contain $$\mathbf{e}_0$$. At inference a one-layer model reads only row 13, which is why the toy script skips the rest. Training uses all fourteen predictions, and a deeper model needs every row as input to the next layer.

### 7.2 Decode as the one-row case

A decode step is the same expression with a single query row:

$$
\underbrace{q_t^\top}_{1 \times 9}\ \underbrace{K^{(m)\top}}_{9 \times m} \to \underbrace{s}_{1 \times m}, \qquad \underbrace{w}_{1 \times m}\ \underbrace{V^{(m)}}_{m \times 9} \to \underbrace{\mathrm{out}}_{1 \times 9},
$$

with no mask, since every cached row lies at or before $$t$$. For the toy, counting query–key products and their multiply-adds,

$$
\begin{array}{l|c|c}
& \text{query–key products} & \text{multiply-adds } (\times\, d = 9)\\ \hline
\text{prefill, masked } 14 \times 14 & 14 \cdot 15 / 2 = 105 & 945\\
\text{decode step } s & 14 + s & 126 \text{ to } 171
\end{array}
$$

In general, attention in prefill costs order $$n^2 d$$ and in one decode step order $$t\,d$$, and both add about $$2P$$ FLOPs per token processed for the projections. Prefill is a few large matrix–matrix products, decode a few matrix–vector products, which is the compute-bound versus bandwidth-bound split of the post.

---

## 8. What it costs: memory

### 8.1 The formula

Each layer stores one key and one value, each of $$d_{\text{head}}$$ elements, per KV head per token:

$$
\text{bytes} = \underbrace{2}_{K,\,V} \cdot \underbrace{L}_{\text{layers}} \cdot \underbrace{H_{kv}}_{\text{KV heads}} \cdot \underbrace{d_{\text{head}}}_{\text{head width}} \cdot \underbrace{S}_{\text{tokens}} \cdot \underbrace{B}_{\text{sequences}} \cdot \underbrace{p}_{\text{bytes per element}}.
$$

For the toy, $$2 \cdot 1 \cdot 1 \cdot 9 \cdot 20 \cdot 1 = 360$$ elements, the count of §5.5.

### 8.2 Llama-2 7B

With $$L = 32$$, $$H_{kv} = H = 32$$, $$d_{\text{head}} = 128$$ and fp16 ($$p = 2$$), one token costs

$$
2 \times 32 \times 32 \times 128 \times 2 = 524{,}288 \text{ bytes} = 512 \text{ KiB} = 0.5 \text{ MiB}.
$$

### 8.3 The table, exactly

$$
\begin{array}{l|c|c|c|r|r|r|r|r}
\text{shape} & L & H_{kv} & d_{\text{head}} & \text{bytes/token} & \text{KiB} & 4\text{K (GiB)} & 8\text{K (GiB)} & 128\text{K (GiB)} \\ \hline
\text{8B-class, GQA} & 32 & 8 & 128 & 131{,}072 & 128 & 0.5 & 1 & 16 \\
\text{8B-class, MHA} & 32 & 32 & 128 & 524{,}288 & 512 & 2 & 4 & 64 \\
\text{70B-class, GQA} & 80 & 8 & 128 & 327{,}680 & 320 & 1.25 & 2.5 & 40 \\
\text{70B-class, MHA} & 80 & 64 & 128 & 2{,}621{,}440 & 2560 & 10 & 20 & 320
\end{array}
$$

Here $$4$$K, $$8$$K and $$128$$K mean $$4{,}096$$, $$8{,}192$$ and $$131{,}072$$ tokens, and the units are binary. The post's "128 KB" per token is $$131{,}072$$ bytes, and "64 GB" at 128K is $$64$$ GiB, which is $$68.72$$ GB.

### 8.4 Cache against weights

Llama-2 7B has $$6{,}738{,}415{,}616$$ fp16 parameters, so its weights occupy $$13{,}476{,}831{,}232$$ bytes, or $$13.48$$ GB. The number of cached tokens that equals them is

$$
\frac{13{,}476{,}831{,}232}{524{,}288} = 25{,}705 \text{ tokens} = 6.28 \times 4{,}096.
$$

With the rounded figure of 14 GB the count is $$26{,}703$$ tokens, $$6.52$$ sequences at 4K. It is exactly $$7 \times 4{,}096 = 28{,}672$$ tokens only if 14 GB is read as 14 GiB. Six to seven 4K sequences, then, carry as much cache as the model has weights.

On an 80 GB accelerator, ignoring activations and allocator overhead, the space left after the Llama-2 7B weights holds

$$
\frac{80 \times 10^9 - 13{,}476{,}831{,}232}{524{,}288} = 126{,}882 \text{ tokens} = 30.98 \times 4{,}096,
$$

that is $$30$$ full sequences at 4K. For the 8B GQA shape, with $$8{,}030{,}261{,}248$$ parameters,

$$
\frac{80 \times 10^9 - 16{,}060{,}522{,}496}{131{,}072} = 487{,}819 \text{ tokens},
$$

which is $$59$$ sequences at 8K but only $$3$$ at 128K.

### 8.5 Linear in context and batch

$$S$$ and $$B$$ enter only through their product. For the 8B GQA shape, one sequence of 32K tokens and four sequences of 8K tokens cost the same:

$$
131{,}072 \times 32{,}768 = 131{,}072 \times 4 \times 8{,}192 = 4{,}294{,}967{,}296 \text{ bytes} = 4 \text{ GiB}.
$$

The reservation must also cover the largest $$S$$ a sequence may reach rather than its current length. §11.1 measures what that costs when memory is reserved contiguously.

---

## 9. What it costs: bandwidth

### 9.1 Bytes read per decode step

Every decode step reads all the weights once, shared across the batch, and every sequence's whole cache:

$$
\text{bytes read per step} = \underbrace{2N_{\text{total}}}_{\text{weights, fp16}} + B \cdot S \cdot \text{bytes per token}.
$$

### 9.2 FLOPs per byte

The forward compute for one token is (Kaplan et al., 2020, Table 1)

$$
C_{\text{forward}} = 2N + 2\,n_{\text{layer}}\,n_{\text{ctx}}\,d_{\text{attn}},
$$

where $$N$$ counts non-embedding parameters and the factor of 2 comes from the multiply-accumulate of each matrix multiplication. For Llama-2 7B, removing the $$32{,}000 \times 4{,}096$$ input embedding and output head,

$$
N = 6{,}738{,}415{,}616 - 2 \cdot 32{,}000 \cdot 4{,}096 = 6{,}476{,}271{,}616,
$$

and with $$n_{\text{layer}} = 32$$ and $$d_{\text{attn}} = 4{,}096$$:

$$
\begin{array}{r|r|r|r}
n_{\text{ctx}} & 2N & 2\,n_{\text{layer}}\,n_{\text{ctx}}\,d_{\text{attn}} & \text{attention share} \\ \hline
1{,}024 & 1.295 \times 10^{10} & 2.68 \times 10^{8} & 2.0\% \\
4{,}096 & 1.295 \times 10^{10} & 10.74 \times 10^{8} & 7.7\% \\
32{,}768 & 1.295 \times 10^{10} & 85.90 \times 10^{8} & 39.9\%
\end{array}
$$

At batch 1 and short context the step computes about $$2N = 1.295 \times 10^{10}$$ FLOPs while moving about $$1.348 \times 10^{10}$$ bytes of weights, so

$$
\frac{\text{FLOPs}}{\text{byte}} \approx \frac{1.295 \times 10^{10}}{1.348 \times 10^{10}} = 0.96 \approx 1,
$$

against the roughly 100 FLOPs per byte the post's Figure 9 puts at the hardware's disposal. The attention term is why FLOPs per token barely move with context at moderate lengths: about $$2\%$$ of the total at 1K and under $$8\%$$ at 4K. At 32K it is about $$40\%$$, so for very long contexts that statement needs qualifying.

### 9.3 The batch crossover

The cache read overtakes the weights when $$B \cdot S \cdot \text{bytes per token}$$ equals the weight bytes. For the 8B GQA shape, taking the weights as 16 GiB,

$$
B \cdot S = \frac{16 \cdot 2^{30}}{131{,}072} = 131{,}072 \text{ tokens} \quad\Longrightarrow\quad B = \frac{131{,}072}{8{,}192} = 16 \text{ at } S = 8\text{K}.
$$

With the actual $$8{,}030{,}261{,}248$$ parameters, $$16.06$$ GB or $$14.96$$ GiB, the crossover is $$B \cdot S = 122{,}532$$, or $$14.96$$ sequences at 8K, the same figure as the GiB count because $$131{,}072 \times 8{,}192 = 2^{30}$$.

$$
\begin{array}{c|r|r|r}
B & S & \text{cache read per step} & \text{cache / weights } (16\ \text{GiB}) \\ \hline
1 & 1{,}024 & 0.125\ \text{GiB} & 0.8\% \\
1 & 8{,}192 & 1\ \text{GiB} & 6.2\% \\
4 & 8{,}192 & 4\ \text{GiB} & 25.0\% \\
16 & 8{,}192 & 16\ \text{GiB} & 100.0\% \\
32 & 8{,}192 & 32\ \text{GiB} & 200.0\%
\end{array}
$$

---

## 10. Making it smaller

### 10.1 Fewer KV heads: GQA and MQA

With $$H$$ query heads and $$H_{kv}$$ KV heads, $$g = H / H_{kv}$$ query heads share each KV head, and query head $$h$$ reads KV head $$\lfloor h / g \rfloor$$. Only $$H_{kv}$$ enters the cache. For Llama-2 70B, with $$L = 80$$, $$H = 64$$, $$d_{\text{head}} = 128$$ and fp16,

$$
\underbrace{2 \cdot 80 \cdot 64 \cdot 128 \cdot 2}_{\text{MHA}} = 2{,}621{,}440 \text{ bytes},
\qquad
\underbrace{2 \cdot 80 \cdot 8 \cdot 128 \cdot 2}_{\text{GQA},\ H_{kv} = 8} = 327{,}680 \text{ bytes},
$$

that is $$2.5$$ MiB against $$0.3125$$ MiB per token, a factor of $$64 / 8 = 8$$.

A worked example with $$H = 4$$ query heads, $$H_{kv} = 2$$ KV heads ($$g = 2$$), $$d_{\text{head}} = 2$$ and three cached tokens, with numbers chosen for readability rather than taken from a model. The cache holds, for KV heads 0 and 1,

$$
K^{[0]} = \begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix}, \quad V^{[0]} = \begin{bmatrix}2&0\\0&2\\1&1\end{bmatrix}, \qquad
K^{[1]} = \begin{bmatrix}2&0\\0&1\\1&-1\end{bmatrix}, \quad V^{[1]} = \begin{bmatrix}1&0\\0&1\\0&0\end{bmatrix}.
$$

The current token's four queries are read against the KV head of their group:

$$
\begin{array}{c|c|c|l|l|l}
h & q^{[h]} & \lfloor h/2 \rfloor & s = K^{[g]} q^{[h]} / \sqrt{2} & w = \mathrm{softmax}(s) & w\,V^{[g]} \\ \hline
0 & (1, 0) & 0 & (0.707, 0.000, 0.707) & (0.401, 0.198, 0.401) & (1.203, 0.797) \\
1 & (0, 1) & 0 & (0.000, 0.707, 0.707) & (0.198, 0.401, 0.401) & (0.797, 1.203) \\
2 & (1, 0) & 1 & (1.414, 0.000, 0.707) & (0.576, 0.140, 0.284) & (0.576, 0.140) \\
3 & (0, 1) & 1 & (0.000, 0.707, -0.707) & (0.284, 0.576, 0.140) & (0.284, 0.576)
\end{array}
$$

Four distinct score patterns come from two sets of cached keys. Per token the cache stores $$2 \times 2 \times 2 = 8$$ numbers, where MHA with four KV heads would store $$2 \times 4 \times 2 = 16$$.

### 10.2 Low-rank compression: MLA

MLA caches, per token, a compressed latent $$c_i \in \mathbb{R}^{d_c}$$ computed from $$h_i$$, plus a small decoupled key of $$d_h^R$$ elements that carries RoPE, and reconstructs the keys and values from $$c_i$$ with learned up-projections. DeepSeek-AI (2024, Table 1) give the cached elements per token as

$$
\text{MHA: } 2\,n_h d_h\,l, \qquad \text{GQA: } 2\,n_g d_h\,l, \qquad \text{MQA: } 2\,d_h\,l, \qquad \text{MLA: } (d_c + d_h^R)\,l \approx \tfrac{9}{2}\,d_h\,l.
$$

With DeepSeek-V2's $$n_h = 128$$, $$d_h = 128$$, $$d_c = 512$$, $$d_h^R = 64$$ and $$l = 60$$,

$$
\text{MHA: } 2 \cdot 128 \cdot 128 = 32{,}768, \qquad \text{MLA: } 512 + 64 = 576 \qquad \text{elements per token per layer},
$$

$$
\frac{32{,}768}{576} = 56.9, \qquad \frac{576}{2 d_h} = \frac{576}{256} = 2.25 \text{ GQA groups},
$$

and over 60 layers at 16 bits, $$3{,}932{,}160$$ bytes ($$3.75$$ MiB) per token for MHA against $$69{,}120$$ bytes ($$67.5$$ KiB) for MLA. Relative to MHA with the same head count, the saving for this configuration is nearly 57-fold.

### 10.3 Fewer bits: quantization

Per token for the 8B GQA shape,

$$
\text{fp16: } 131{,}072, \qquad \text{fp8: } 65{,}536, \qquad \text{int4: } 32{,}768 \text{ bytes},
$$

before the few bytes of scale metadata each scheme adds. Symmetric per-vector quantization to $$b$$ bits maps a vector $$k$$ to integers with a single scale:

$$
s = \frac{\max_j \lvert k_j \rvert}{2^{b-1} - 1}, \qquad \hat{q}_j = \mathrm{clip}\big(\mathrm{round}(k_j / s),\ -2^{b-1},\ 2^{b-1} - 1\big), \qquad \tilde{k}_j = s\,\hat{q}_j,
$$

so each element's error is at most $$s/2$$. Applied to the demo's layer-1 key at position 2 from §6.4,

$$
k = \begin{bmatrix}0.3848 & 0.0690 & -0.2583 & 0.0087 & -0.1501 & -0.4039 & -0.1339 & -0.0287\end{bmatrix}
$$

with int8 ($$b = 8$$),

$$
s = 0.00318018, \qquad \hat{q} = (121, 22, -81, 3, -47, -127, -42, -9), \qquad \max_j \lvert \tilde{k}_j - k_j \rvert = 9.69\times 10^{-4} \le \frac{s}{2} = 1.59\times 10^{-3},
$$

and with int4 ($$b = 4$$),

$$
s = 0.05769759, \qquad \hat{q} = (7, 1, -4, 0, -3, -7, -2, 0), \qquad \max_j \lvert \tilde{k}_j - k_j \rvert = 2.87\times 10^{-2}.
$$

The score this key contributes, $$q_2^{(1)} k^\top / \sqrt{8}$$, moves from $$0.00768$$ to $$0.00762$$ under int8 and to $$0.00277$$ under int4.

### 10.4 Fewer tokens: windows and sinks

A window of $$W$$ tokens caps each layer's cache at $$W$$ rows whatever $$S$$ is:

$$
\text{bytes} \le 2 \cdot L \cdot H_{kv} \cdot d_{\text{head}} \cdot W \cdot B \cdot p, \qquad \text{8B GQA, } W = 4{,}096: \quad 4{,}096 \times 131{,}072 \text{ bytes} = 512 \text{ MiB per sequence}.
$$

The toy shows what a window costs. At step 0 the query $$q_{13}$$ reads only rows $$14 - W, \dots, 13$$, and row 6 survives only if $$14 - W \le 6$$, that is $$W \ge 8$$:

$$
\begin{array}{c|c|l|c|c}
W & \text{rows kept} & \text{top token} & p & p(\texttt{woof}) \\ \hline
2 & 12\text{–}13 & \texttt{->} & 0.9820 & 0.0000 \\
4 & 10\text{–}13 & \texttt{->} & 0.3313 & 0.0000 \\
6 & 8\text{–}13 & \texttt{->} & 0.3333 & 0.0000 \\
7 & 7\text{–}13 & \texttt{;} & 0.4932 & 0.0000 \\
8 & 6\text{–}13 & \texttt{woof} & 0.9309 & 0.9309 \\
10 & 4\text{–}13 & \texttt{woof} & 0.9150 & 0.9150 \\
14 & 0\text{–}13 & \texttt{woof} & 0.8848 & 0.8848
\end{array}
$$

Keeping the first two rows as attention sinks does not rescue it:

$$
\begin{array}{c|c|l|c|c}
\text{sinks} + W & \text{rows kept} & \text{top token} & p & p(\texttt{woof}) \\ \hline
2 + 4 & \{0, 1\} \cup \{10,\dots,13\} & \texttt{->} & 0.3333 & 0.0000 \\
2 + 6 & \{0, 1\} \cup \{8,\dots,13\} & \texttt{->} & 0.3353 & 0.0000
\end{array}
$$

The toy's answer sits in the middle of the sequence, which is exactly what a window evicts. Sinks help trained models for a different reason: those models park surplus attention on the first positions, and evicting them distorts every softmax. A pure lookup like the toy has no such surplus to protect.

---

## 11. Managing it in a real server

### 11.1 Contiguous versus paged

A request that ends at 100 tokens inside a 4,096-token contiguous reservation leaves $$4{,}096 - 100 = 3{,}996$$ slots unused, $$97.6\%$$ of the reservation. Paged into blocks of 16,

$$
\left\lceil \frac{100}{16} \right\rceil = 7 \text{ blocks}, \qquad 7 \times 16 = 112 \text{ slots}, \qquad 112 - 100 = 12 \text{ unused},
$$

and paged waste per sequence never exceeds $$15$$ slots. At $$131{,}072$$ bytes per slot (8B GQA), that is $$512$$ MiB reserved contiguously, $$12.5$$ MiB actually used, and $$14$$ MiB allocated when paged.

### 11.2 Addressing through a block table

A sequence's block table $$\mathrm{BT}$$ maps logical block numbers to physical blocks, and the physical slot of logical position $$i$$ is

$$
\mathrm{slot}(i) = 16 \cdot \mathrm{BT}\big[\lfloor i / 16 \rfloor\big] + (i \bmod 16).
$$

With the illustrative table $$\mathrm{BT} = [5, 2, 9]$$, position 37 lies in logical block $$\lfloor 37 / 16 \rfloor = 2$$ at offset $$37 \bmod 16 = 5$$, so

$$
\mathrm{slot}(37) = 16 \cdot 9 + 5 = 149.
$$

Positions that are contiguous in the sequence are scattered in memory, and the attention kernel gathers them through $$\mathrm{BT}$$.

### 11.3 Sharing and copy-on-write

Two sequences that begin with the same 40 tokens have identical keys and values at positions 0–39, by §3.2. Positions 0–31 fill two complete blocks, which both block tables can point to:

$$
\mathrm{BT}_A = [5, 2, 9], \qquad \mathrm{BT}_B = [5, 2, 11], \qquad \mathrm{refcount}(5) = \mathrm{refcount}(2) = 2.
$$

Positions 32–39 occupy 8 slots of a third block that each sequence holds separately, because each sequence's next token lands in that block. With $$B$$ sequences sharing the prefix, the two shared blocks save $$32\,(B - 1)$$ slots. A forked beam is the same situation: it shares every block with its parent and copies a partially filled block the first time it writes to it.

---

## 12. What the cache does not do

### 12.1 It does not fix the sampled text

Two requests with the same cached prefix obtain the same logits and then sample independently. With the toy's step-0 distribution $$p$$ and plain sampling at temperature 1, the probability that two draws agree is

$$
\sum_j p_j^2 = 0.7868,
$$

so two such requests begin with different tokens about $$21.3\%$$ of the time.

### 12.2 It is not shared across models

A cached row is $$k_i = h_i W_K$$ for one model's $$W_K$$ and one model's $$h_i$$. A model with different weights produces different rows from the same tokens, and a quantized cache (§10.3) holds different numbers again, so no model can read another's cache.

---

## 13. Summary of every expression

$$
\begin{aligned}
&\text{cache} && K^{(m)},\ V^{(m)} \in \mathbb{R}^{m \times d}\\
&\text{append} && K^{(m+1)} = \begin{bmatrix} K^{(m)} \\ k_m^\top \end{bmatrix}, \quad V^{(m+1)} = \begin{bmatrix} V^{(m)} \\ v_m^\top \end{bmatrix}\\
&\text{decode read} && \mathrm{out}_t = \mathrm{softmax}\big(\lambda\, q_t^\top K^{(t+1)\top}\big)\, V^{(t+1)}\\
&\text{prefill} && \mathrm{OUT} = \mathrm{softmax}_{\text{row}}\big(\lambda\, Q K^\top + M\big)\, V\\
&\text{invariant} && k_i^{(\ell)},\ v_i^{(\ell)} \text{ depend only on } x_0, \dots, x_i\\
&\text{projections} && T n_0 + T(T-1)/2 \ \text{ vs } \ n_0 + T\\
&\text{score ops} && L \textstyle\sum_{s} (n_0+s)(n_0+s+1)/2 \ \text{ vs } \ L\,(n_0+T)(n_0+T+1)/2\\
&\text{memory} && 2 \cdot L \cdot H_{kv} \cdot d_{\text{head}} \cdot S \cdot B \cdot p\\
&\text{bandwidth} && 2N_{\text{total}} + B \cdot S \cdot \text{bytes per token, per step}\\
&\text{compute} && C_{\text{forward}} = 2N + 2\,n_{\text{layer}}\,n_{\text{ctx}}\,d_{\text{attn}}\\
&\text{GQA} && \text{query head } h \text{ reads KV head } \lfloor h/g \rfloor, \quad g = H / H_{kv}\\
&\text{MLA} && 2\,n_h d_h\,l \ \to\ (d_c + d_h^R)\,l\\
&\text{quantization} && s = \max_j \lvert k_j \rvert / (2^{b-1} - 1), \quad \lvert \tilde{k}_j - k_j \rvert \le s/2\\
&\text{window} && \text{rows per layer} \le W\\
&\text{paging} && \mathrm{slot}(i) = 16\,\mathrm{BT}[\lfloor i/16 \rfloor] + (i \bmod 16)
\end{aligned}
$$

---

## References

- [Vaswani et al. (2017), *Attention Is All You Need*](https://arxiv.org/abs/1706.03762)
- [Kaplan et al. (2020), *Scaling Laws for Neural Language Models*](https://arxiv.org/abs/2001.08361)
- [Su et al. (2021), *RoFormer: Enhanced Transformer with Rotary Position Embedding*](https://arxiv.org/abs/2104.09864)
- [Pope et al. (2022), *Efficiently Scaling Transformer Inference*](https://arxiv.org/abs/2211.05102)
- [Shazeer (2019), *Fast Transformer Decoding: One Write-Head is All You Need*](https://arxiv.org/abs/1911.02150)
- [Ainslie et al. (2023), *GQA: Training Generalized Multi-Query Transformer Models*](https://arxiv.org/abs/2305.13245)
- [Kwon et al. (2023), *Efficient Memory Management for LLM Serving with PagedAttention*](https://arxiv.org/abs/2309.06180)
- [Xiao et al. (2023), *Efficient Streaming Language Models with Attention Sinks*](https://arxiv.org/abs/2309.17453)
- [DeepSeek-AI (2024), *DeepSeek-V2 technical report*](https://arxiv.org/abs/2405.04434)

*Sources: [`toy_llm_demo.py`](https://gist.github.com/chuan2019/c9b632faeb9422ca2680af9ca6066dad) and [`kv_cache_demo.py`](https://gist.github.com/chuan2019/f2bf188647210f39a7c987dca6be62bf), dependency-free Python. Every number in this document was produced by running them or by the arithmetic shown.*
