---
title: "Attention by Hand: A Complete Mathematical Walkthrough of a Toy Transformer"
date: 2026-09-12 10:00:00 +0000
categories: [Machine Learning]
tags: [transformers, attention, kv-cache, llm, linear-algebra, softmax, backpropagation]
description: "Every quantity in a nine-token, one-head, one-layer transformer derived end to end in vector and matrix form — embedding, keys, scores, softmax, output, generation, and the gradients — with all intermediate numbers filled in."
math: true
---

A transformer small enough that every number it produces can be read directly is the fastest way to see what attention actually computes. This post derives every quantity in such a model end to end, in vector and matrix form, with all intermediate values filled in.

It is the mathematical companion to [Part 1 of *From Attention to Prompt Caching*](https://medium.com/@chuan-zhang/from-attention-to-prompt-caching-a-trilogy-1-9481e88d2a87), which introduces the same toy model in prose. All numbers below are reproduced from [`toy_llm_demo.py`](https://gist.github.com/chuan2019/c9b632faeb9422ca2680af9ca6066dad) — dependency-free Python, standard library only.

---

## 1. Notation and setup

### 1.1 Vocabulary

The prompt is split on whitespace, and the vocabulary is built in order of first appearance with the sentinel $$\texttt{<bos>}$$ seeded at index $$0$$:

$$
\mathcal{V} = \{\,\texttt{<bos>},\ \texttt{cat},\ \texttt{->},\ \texttt{meow},\ \texttt{;},\ \texttt{dog},\ \texttt{woof},\ \texttt{cow},\ \texttt{moo}\,\}
$$

$$
\begin{array}{c|ccccccccc}
\text{token} & \texttt{<bos>} & \texttt{cat} & \texttt{->} & \texttt{meow} & \texttt{;} & \texttt{dog} & \texttt{woof} & \texttt{cow} & \texttt{moo}\\ \hline
\text{index} & 0 & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8
\end{array}
$$

So $$\lvert\mathcal{V}\rvert = 9$$, and the model sets

$$
d_{\text{model}} = |\mathcal{V}| = 9, \qquad H = 1, \qquad d_{\text{head}} = d_{\text{model}} = 9, \qquad L = 1 .
$$

### 1.2 The token sequence

$$
\texttt{cat -> meow ; dog -> woof ; cow -> moo ; dog ->}
$$

gives $$n = 14$$ tokens. Writing $$x_i$$ for the token at position $$i$$ and $$\mathrm{id}(x_i)$$ for its index:

$$
\big(\mathrm{id}(x_0),\dots,\mathrm{id}(x_{13})\big) = (1,\,2,\,3,\,4,\,5,\,2,\,6,\,4,\,7,\,2,\,8,\,4,\,5,\,2)
$$

### 1.3 Boundary convention

Positions before the start of the sequence return the sentinel:

$$
x_{i} := \texttt{<bos>} \quad \text{for } i < 0 .
$$

This is what keeps the key rule total at $$i = 0$$ and $$i = 1$$, and it is the source of the only entry in the model that holds a $$2$$.

---

## 2. Step 1 — Embedding

### 2.1 The one-hot map

Let $$\mathbf{e}_j \in \mathbb{R}^{9}$$ be the $$j$$-th standard basis vector. The embedding is the identity on indices:

$$
e(x) = \mathbf{e}_{\mathrm{id}(x)}, \qquad e : \mathcal{V} \to \mathbb{R}^{9}
$$

Equivalently, the embedding matrix is $$W_E = I_9$$.

### 2.2 The sequence matrix

Stack the embeddings as rows to form $$X \in \mathbb{R}^{14 \times 9}$$, where $$X_{i,j} = 1$$ iff $$\mathrm{id}(x_i) = j$$:

$$
X=
\begin{bmatrix}
0&1&0&0&0&0&0&0&0\\
0&0&1&0&0&0&0&0&0\\
0&0&0&1&0&0&0&0&0\\
0&0&0&0&1&0&0&0&0\\
0&0&0&0&0&1&0&0&0\\
0&0&1&0&0&0&0&0&0\\
0&0&0&0&0&0&1&0&0\\
0&0&0&0&1&0&0&0&0\\
0&0&0&0&0&0&0&1&0\\
0&0&1&0&0&0&0&0&0\\
0&0&0&0&0&0&0&0&1\\
0&0&0&0&1&0&0&0&0\\
0&0&0&0&0&1&0&0&0\\
0&0&1&0&0&0&0&0&0
\end{bmatrix}
$$

Two properties of one-hot vectors are used throughout:

- **Sums stay legible.** $$e(a) + e(b)$$ has exactly two ones (or a single $$2$$ when $$a = b$$), and no other unordered pair produces the same vector.
- **Dot products count matches.** For $$u, v \in \{0,1\}^{9}$$, $$\;u \cdot v = \big\lvert\{\,j : u_j = v_j = 1\,\}\big\rvert$$.

---

## 3. Step 2 — The three projections

### 3.1 The rules

$$
\begin{aligned}
k_i &= e(x_{i-2}) + e(x_{i-1}) && \text{the two tokens before position } i\\
v_i &= e(x_i) && \text{the token at position } i\\
q &= e(x_{t-1}) + e(x_t) && \text{the last two tokens, } t = n-1
\end{aligned}
$$

Read as a data structure: $$v_i$$ is the **payload**, $$k_i$$ is the **address** it was filed under, and $$q$$ is the **address being looked up**. The key deliberately excludes $$x_i$$ itself; if the address contained the answer, the answer would be needed to find it.

### 3.2 Matrix form on the position axis

Let $$\bar{X} \in \mathbb{R}^{16 \times 9}$$ be $$X$$ with two $$\texttt{<bos>}$$ rows prepended, indexed $$i = -2,\dots,13$$. Let $$P$$ be the down-shift operator, $$(P M)_i = M_{i-1}$$. Then

$$
K = \big[(P + P^{2})\,\bar{X}\big]_{0:14} \in \mathbb{R}^{14 \times 9}, \qquad V = X \in \mathbb{R}^{14 \times 9}
$$

and the query is the last row of $$(P + P^0)\bar X$$, i.e. $$q^\top = \bar X_{t-1} + \bar X_{t}$$.

This is the structural point of the toy: $$P$$ acts on the **position** axis. A real $$W_K \in \mathbb{R}^{d_{\text{head}} \times d_{\text{model}}}$$ acts on the **feature** axis and sees only the residual stream at position $$i$$, so it cannot reach backwards to $$i-1$$. Since attention is the only operation that moves information between positions, a real model **cannot express this rule in a single layer**; it requires the two-layer induction circuit of §9.

### 3.3 The key matrix, numerically

Applying §3.1 position by position:

$$
\begin{array}{c|l|l}
i & k_i \text{ (symbolic)} & v_i\\ \hline
0 & e(\texttt{<bos>}) + e(\texttt{<bos>}) = 2\,\mathbf{e}_0 & e(\texttt{cat})\\
1 & e(\texttt{<bos>}) + e(\texttt{cat}) & e(\texttt{->})\\
2 & e(\texttt{cat}) + e(\texttt{->}) & e(\texttt{meow})\\
3 & e(\texttt{->}) + e(\texttt{meow}) & e(\texttt{;})\\
4 & e(\texttt{meow}) + e(\texttt{;}) & e(\texttt{dog})\\
5 & e(\texttt{;}) + e(\texttt{dog}) & e(\texttt{->})\\
6 & e(\texttt{dog}) + e(\texttt{->}) & e(\texttt{woof})\\
7 & e(\texttt{->}) + e(\texttt{woof}) & e(\texttt{;})\\
8 & e(\texttt{woof}) + e(\texttt{;}) & e(\texttt{cow})\\
9 & e(\texttt{;}) + e(\texttt{cow}) & e(\texttt{->})\\
10 & e(\texttt{cow}) + e(\texttt{->}) & e(\texttt{moo})\\
11 & e(\texttt{->}) + e(\texttt{moo}) & e(\texttt{;})\\
12 & e(\texttt{moo}) + e(\texttt{;}) & e(\texttt{dog})\\
13 & e(\texttt{;}) + e(\texttt{dog}) & e(\texttt{->})
\end{array}
$$

As a matrix, with columns ordered $$(\texttt{<bos>},\texttt{cat},\texttt{->},\texttt{meow},\texttt{;},\texttt{dog},\texttt{woof},\texttt{cow},\texttt{moo})$$:

$$
K=
\begin{bmatrix}
2&0&0&0&0&0&0&0&0\\
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
0&0&0&0&1&1&0&0&0
\end{bmatrix}
$$

Note $$K_0 = 2\,\mathbf{e}_0$$: both padding slots are $$\texttt{<bos>}$$, so the two basis vectors land in the same column. It is the only entry in the model holding a $$2$$.

Note also $$K_5 = K_{13} = \mathbf{e}_4 + \mathbf{e}_5$$. Identical context at different positions yields an identical key, because the toy has **no positional encoding**. In a real model, where RoPE ([Su et al., 2021](https://arxiv.org/pdf/2104.09864)) or ALiBi ([Press, Smith & Lewis, 2021](https://arxiv.org/pdf/2108.12409)) folds position into $$K$$, these rows would differ — the constraint that Part 3 lives inside.

### 3.4 The query

With $$t = 13$$, $$x_{12} = \texttt{dog}$$ and $$x_{13} = \texttt{->}$$:

$$
q = e(\texttt{dog}) + e(\texttt{->}) = \mathbf{e}_5 + \mathbf{e}_2 = \begin{bmatrix}0&0&1&0&0&1&0&0&0\end{bmatrix}^\top
$$

### 3.5 Why a two-token window

$$\texttt{->}$$ occurs four times in the prompt, at positions $$1, 5, 9, 13$$. The three occurrences with a successor are followed by $$\texttt{meow}$$, $$\texttt{woof}$$, $$\texttt{moo}$$ respectively. A one-token window therefore cannot discriminate:

$$
\big|\{\,x_{i+1} : x_i = \texttt{->}\,\}\big| = 3 \quad\Longrightarrow\quad \text{ambiguous}
$$

Two tokens is the shortest window that makes the lookup unique.

---

## 4. Step 3 — Scoring

### 4.1 Definition

$$
s_i = \lambda \,(q \cdot k_i), \qquad \lambda = 4
$$

or in matrix form, as a row vector,

$$
s = \lambda\, q^\top K^\top \in \mathbb{R}^{1 \times 14}
$$

$$\lambda$$ is a sharpness knob standing in for the $$1/\sqrt{d}$$ of a real model. It does not change **which** key wins, only how decisively:

$$
\arg\max_i \lambda (q\cdot k_i) = \arg\max_i (q \cdot k_i) \quad \text{for } \lambda > 0
$$

### 4.2 The dot products

Because both $$q$$ and every $$k_i$$ are $$0/1$$ vectors, $$q \cdot k_i$$ is a literal match count. With $$q = \mathbf{e}_2 + \mathbf{e}_5$$:

$$
q^\top K^\top = \begin{bmatrix}0&0&1&1&0&1&\mathbf{2}&1&0&0&1&1&0&1\end{bmatrix}
$$

$$
s = 4 \cdot q^\top K^\top = \begin{bmatrix}0&0&4&4&0&4&\mathbf{8}&4&0&0&4&4&0&4\end{bmatrix}
$$

The score vector takes exactly three values:

$$
\begin{array}{c|c|l|c}
q\cdot k_i & s_i & \text{positions} & \text{count}\\ \hline
2 & 8 & \{6\} & 1\\
1 & 4 & \{2,3,5,7,10,11,13\} & 7\\
0 & 0 & \{0,1,4,8,9,12\} & 6
\end{array}
$$

Position $$6$$ is the unique row whose key matches **both** query columns: $$k_6 = e(\texttt{dog}) + e(\texttt{->})$$, and $$v_6 = e(\texttt{woof})$$.

---

## 5. Step 4 — Softmax

### 5.1 Definition

$$
w_i = \frac{\exp(s_i - \max_j s_j)}{\sum_{j=0}^{13} \exp(s_j - \max_j s_j)}
$$

Subtracting $$\max_j s_j$$ guards against overflow and cancels exactly:

$$
\frac{e^{s_i - m}}{\sum_j e^{s_j - m}} = \frac{e^{-m} e^{s_i}}{e^{-m}\sum_j e^{s_j}} = \frac{e^{s_i}}{\sum_j e^{s_j}}
$$

### 5.2 The arithmetic

Here $$m = 8$$, so the three distinct exponentials are

$$
e^{0} = 1, \qquad e^{-4} = 0.018315639, \qquad e^{-8} = 0.000335463
$$

The normalizer collects one, seven and six of them respectively:

$$
\begin{aligned}
Z &= 1\cdot e^{0} + 7 \cdot e^{-4} + 6 \cdot e^{-8}\\
  &= 1 + 7(0.018315639) + 6(0.000335463)\\
  &= 1 + 0.128209473 + 0.002012778\\
  &= 1.130222251
\end{aligned}
$$

giving the three attention weights

$$
\begin{aligned}
w_6 &= \frac{1}{1.130222251} = 0.884781734\\[4pt]
w_i &= \frac{0.018315639}{1.130222251} = 0.016205343, && i \in \{2,3,5,7,10,11,13\}\\[4pt]
w_i &= \frac{0.000335463}{1.130222251} = 0.000296811, && i \in \{0,1,4,8,9,12\}
\end{aligned}
$$

Check: $$0.884781734 + 7(0.016205343) + 6(0.000296811) = 1.000000000$$.

As a vector,

$$
w = \begin{bmatrix}
0.000297 & 0.000297 & 0.016205 & 0.016205 & 0.000297 & 0.016205 & \mathbf{0.884782} \\
0.016205 & 0.000297 & 0.000297 & 0.016205 & 0.016205 & 0.000297 & 0.016205
\end{bmatrix}
$$

(read left to right, top row positions $$0$$–$$6$$, bottom row positions $$7$$–$$13$$).

---

## 6. Step 5 — Output

### 6.1 The weighted sum

$$
\mathrm{out} = \sum_{i=0}^{13} w_i\, v_i = w^\top V \in \mathbb{R}^{9}
$$

### 6.2 Why there is no LM head

Because every $$v_i = \mathbf{e}_{\mathrm{id}(x_i)}$$ is a basis vector, the weighted sum is a **scatter-add** into vocabulary slots:

$$
\mathrm{out}_j = \sum_{i \,:\, \mathrm{id}(x_i) = j} w_i
$$

The result is therefore already indexed by vocabulary and already normalized, since $$\sum_j \mathrm{out}_j = \sum_i w_i = 1$$. A real model's output projection $$W_U \in \mathbb{R}^{d_{\text{model}} \times \lvert\mathcal{V}\rvert}$$ and its closing softmax would both have nothing left to do. Formally, the toy has $$W_U = I_9$$ and the softmax over the $$14$$ scores **is** the output distribution.

### 6.3 The distribution, term by term

$$
\begin{aligned}
p(\texttt{woof}) &= w_6 &&= 0.884781734\\
p(\texttt{;})    &= w_3 + w_7 + w_{11} &&= 3(0.016205343) = 0.048616028\\
p(\texttt{->})   &= w_1 + w_5 + w_9 + w_{13} &&= 2(0.000296811) + 2(0.016205343) = 0.033004308\\
p(\texttt{meow}) &= w_2 &&= 0.016205343\\
p(\texttt{moo})  &= w_{10} &&= 0.016205343\\
p(\texttt{dog})  &= w_4 + w_{12} &&= 2(0.000296811) = 0.000593622\\
p(\texttt{cat})  &= w_0 &&= 0.000296811\\
p(\texttt{cow})  &= w_8 &&= 0.000296811\\
p(\texttt{<bos>}) &= 0 &&= 0
\end{aligned}
$$

with $$\sum_j p_j = 1$$.

Two structural facts are visible in these numbers:

- $$p(\texttt{<bos>}) = 0$$ **exactly, at every step**, because no position ever holds $$\texttt{<bos>}$$ as its own token, so $$\texttt{<bos>}$$ is never a value. One column of the distribution is empty by construction.
- $$p(\texttt{;}) = 0.0486$$ exceeds $$p(\texttt{meow}) = 0.0162$$ even though every contributing position carries the same weight. The runner-up is assembled from **three** positions at $$0.0162$$ each. A token weakly supported by many positions outscores one weakly supported by a single position — the weighted-sum-of-rows behaviour made visible.

---

## 7. The whole step as one expression

Steps 2 through 4 collapse into a single line:

$$
\boxed{\;\mathrm{out} = \mathrm{softmax}\!\big(\lambda \, q^\top K^\top\big)\,V\;}
\qquad\text{(the toy, one decode step)}
$$

$$
\mathrm{out} = \mathrm{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_{\text{head}}}} + M\right) V
\qquad\text{(a real model, per head, per layer)}
$$

### 7.1 Shapes

$$
\underbrace{q^\top}_{1 \times 9}\;\underbrace{K^\top}_{9 \times 14} \;\longrightarrow\; \underbrace{s}_{1 \times 14} \;\longrightarrow\; \underbrace{w}_{1\times 14}\;\underbrace{V}_{14 \times 9} \;\longrightarrow\; \underbrace{\mathrm{out}}_{1 \times 9}
$$

### 7.2 The axis flip

$$
\underbrace{q^\top K^\top}_{\text{contracts the \textit{feature} axis}} \;\Rightarrow\; \text{indexed by position}, \qquad
\underbrace{w V}_{\text{contracts the \textit{position} axis}} \;\Rightarrow\; \text{indexed by features}
$$

The intermediate object $$s$$ is the only one carrying two token axes in the general ($$n$$ query rows) case, and that is exactly where the $$O(n^2)$$ cost of attention lives.

### 7.3 The causal mask

The toy forms a single query row, so ordering holds by construction and $$M$$ is absent as a tensor. A real model processing a prompt of length $$n$$ at once forms $$n$$ query rows and needs

$$
M \in \mathbb{R}^{n \times n}, \qquad M_{ij} = \begin{cases} 0 & j \le i\\ -\infty & j > i\end{cases}
$$

so that $$\mathrm{softmax}$$ assigns exactly zero weight to every future key:

$$
\lim_{M_{ij} \to -\infty} \frac{e^{s_{ij} + M_{ij}}}{\sum_l e^{s_{il} + M_{il}}} = 0 \quad \text{for } j > i
$$

---

## 8. Step 6 — Autoregressive generation

Sampling is greedy: $$x_{t+1} = \arg\max_j \mathrm{out}_j$$, appended to the sequence. Each generated token then enters both the key matrix and the next query, so $$K$$ gains one row per step and $$q$$ is rebuilt from the new final pair.

### 8.1 Recurrence

$$
\begin{aligned}
k_{t+1} &= e(x_{t-1}) + e(x_{t}) & &\text{(one new row appended to } K)\\
v_{t+1} &= e(x_{t+1}) & &\text{(one new row appended to } V)\\
q^{(t+1)} &= e(x_{t}) + e(x_{t+1}) & &\text{(rebuilt, then discarded)}
\end{aligned}
$$

Note that $$k_{t+1}$$ and $$v_{t+1}$$ depend only on tokens at or before $$t+1$$, and every earlier row is untouched. That is the cache invariant developed in Part 2.

### 8.2 The six steps

$$
\begin{array}{c|l|c|c|l|c}
\text{step} & q & \arg\max & p & \text{runner-up} & p\\ \hline
0 & e(\texttt{dog}) + e(\texttt{->})   & \texttt{woof} & 0.884782 & \texttt{;}    & 0.048616\\
1 & e(\texttt{->}) + e(\texttt{woof})  & \texttt{;}    & 0.916920 & \texttt{woof} & 0.032401\\
2 & e(\texttt{woof}) + e(\texttt{;})   & \texttt{cow}  & 0.884257 & \texttt{->}   & 0.048884\\
3 & e(\texttt{;}) + e(\texttt{cow})    & \texttt{->}   & 0.916673 & \texttt{dog}  & 0.032382\\
4 & e(\texttt{cow}) + e(\texttt{->})   & \texttt{moo}  & 0.856513 & \texttt{;}    & 0.062750\\
5 & e(\texttt{->}) + e(\texttt{moo})   & \texttt{;}    & 0.903316 & \texttt{woof} & 0.031366
\end{array}
$$

$$
\text{prompt: } \texttt{cat -> meow ; dog -> woof ; cow -> moo ; dog ->}
$$

$$
\text{response: } \texttt{woof ; cow -> moo ;}
$$

### 8.3 Worked detail for step 1

After appending $$\texttt{woof}$$ at position $$14$$, the sequence has length $$15$$ and

$$
q^{(1)} = e(\texttt{->}) + e(\texttt{woof}) = \mathbf{e}_2 + \mathbf{e}_6, \qquad k_{14} = e(\texttt{dog}) + e(\texttt{->}) = \mathbf{e}_5 + \mathbf{e}_2
$$

$$
q^{(1)\top} K^\top = \begin{bmatrix}0&0&1&1&0&0&1&\mathbf{2}&1&0&1&1&0&0&1\end{bmatrix}
$$

so the score distribution is now one $$8$$ (at position $$7$$), seven $$4$$s and seven $$0$$s:

$$
Z = 1 + 7(0.018315639) + 7(0.000335463) = 1.130557711
$$

$$
w_7 = \frac{1}{1.130557711} = 0.884519198, \qquad w^{(4)} = 0.016200534, \qquad w^{(0)} = 0.000296723
$$

Position $$7$$ holds $$v_7 = e(\texttt{;})$$, and positions $$3$$ and $$11$$ also hold $$\texttt{;}$$ with weight $$0.016200534$$ each, so

$$
p(\texttt{;}) = 0.884519198 + 2(0.016200534) = 0.916920267
$$

$$
p(\texttt{woof}) = w_6 + w_{14} = 2(0.016200534) = 0.032401068
$$

confirming row 1 of the table.

### 8.4 The ceiling

The first token is the whole of the model's competence. Thereafter the query is always built from two tokens that already occurred as a consecutive pair, so the lookup walks the cycle the prompt describes:

$$
\texttt{dog}\,\texttt{->} \mapsto \texttt{woof} \mapsto \texttt{;} \mapsto \texttt{cow} \mapsto \texttt{->} \mapsto \texttt{moo} \mapsto \texttt{;} \mapsto \cdots
$$

Note the drift in the peak probability across steps, from $$0.8848$$ down to $$0.8565$$ at step 4. Nothing is degrading. The denominator grows as positions accumulate:

$$
Z^{(t)} = 1 + n_1^{(t)} e^{-4} + n_0^{(t)} e^{-8}, \qquad n_1^{(t)} + n_0^{(t)} + 1 = 14 + t
$$

At step 4 the number of partial matches rises from $$7$$ to $$9$$, so $$Z$$ rises from $$1.1303$$ to $$1.1675$$ and every weight is scaled down accordingly. Longer context dilutes a fixed-strength match — a toy-scale preview of why attention entropy grows with sequence length.

### 8.5 Sampling in a real model

Greedy decoding is the degenerate case. A real sampler reshapes the distribution first, with temperature $$T$$, then nucleus or top-$$k$$ truncation:

$$
p^{(T)}_j = \frac{\exp(\ell_j / T)}{\sum_l \exp(\ell_l / T)}, \qquad T \to 0 \;\Rightarrow\; \arg\max
$$

This is a property of the sampler, strictly downstream of everything above, which is why the same prompt can return different text twice with an identical cache. It is the single most common misconception about prompt caching, and Part 3 returns to it.

---

## 9. Generalization to a real model

### 9.1 Multi-head attention

A real layer projects once at full width and **partitions** the result, rather than running $$H$$ copies of a $$d_{\text{model}}$$-wide computation:

$$
d_{\text{head}} = \frac{d_{\text{model}}}{H}
$$

$$
Q = X W_Q, \quad K = X W_K, \quad V = X W_V, \qquad W_\bullet \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}
$$

Read as $$H$$ contiguous slices $$Q_h, K_h, V_h \in \mathbb{R}^{n \times d_{\text{head}}}$$, each head computes independently:

$$
\mathrm{head}_h = \mathrm{softmax}\!\left(\frac{Q_h K_h^\top}{\sqrt{d_{\text{head}}}} + M\right) V_h \in \mathbb{R}^{n \times d_{\text{head}}}
$$

$$
\mathrm{MHA}(X) = \big[\,\mathrm{head}_1 \;\|\; \cdots \;\|\; \mathrm{head}_H\,\big]\, W_O, \qquad W_O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}
$$

Nothing crosses between heads until $$W_O$$. The cost is therefore about that of one $$d_{\text{model}}$$-wide head; the gain is specialization, not capacity.

### 9.2 The consequence for stored state

Under plain MHA the state stored per token per layer is

$$
2 \times H \times d_{\text{head}} = 2\,d_{\text{model}} \text{ floats}
$$

and over $$L$$ layers, batch $$B$$, sequence length $$S$$, at $$p$$ bytes per element:

$$
\text{bytes} = 2 \cdot L \cdot H_{kv} \cdot d_{\text{head}} \cdot S \cdot B \cdot p
$$

Because heads never read each other's scores, several query heads may share one $$K/V$$ pair, replacing $$H$$ with $$H_{kv} < H$$. That is grouped-query attention, and it is the subject of Part 2.

### 9.3 Why the scaling is $$1/\sqrt{d}$$

For $$q, k \in \mathbb{R}^{d}$$ with independent zero-mean unit-variance components,

$$
\mathbb{E}[q \cdot k] = 0, \qquad \mathrm{Var}(q \cdot k) = \sum_{i=1}^{d}\mathrm{Var}(q_i k_i) = d
$$

so $$q \cdot k$$ has standard deviation $$\sqrt{d}$$ and grows with width. Dividing by $$\sqrt{d}$$ restores unit variance and keeps the softmax out of saturation, where its gradient would vanish. The toy's $$\lambda = 4$$ plays the opposite role deliberately, sharpening rather than flattening.

---

## 10. Training

### 10.1 The objective

There is no data that trains $$W_K$$ and no data that trains $$W_V$$. There is one scalar:

$$
\mathcal{L} = -\sum_{t} \log p\big(x_{t+1} \mid x_{\le t}\big)
$$

Next-token cross-entropy, averaged over every position of every sequence. The labels are free, since the target at position $$t$$ is the token at $$t+1$$, and causal masking yields $$n$$ training targets from one forward pass over an $$n$$-token sequence.

### 10.2 Gradients through attention

Write $$g = \partial \mathcal{L} / \partial\, \mathrm{out}$$ for the gradient arriving at the block output. Since $$\mathrm{out} = \sum_i w_i v_i$$:

$$
\frac{\partial \mathcal{L}}{\partial v_i} = w_i\, g
$$

Gradient reaches a value in proportion to how much that position was attended to. A value nobody looked at barely moves.

For the weights, $$\partial \mathcal{L} / \partial w_i = g \cdot v_i$$, and the softmax Jacobian is

$$
\frac{\partial w_i}{\partial s_j} = w_i\big(\delta_{ij} - w_j\big)
$$

Composing them:

$$
\frac{\partial \mathcal{L}}{\partial s_i} = \sum_j \frac{\partial \mathcal{L}}{\partial w_j}\frac{\partial w_j}{\partial s_i} = \sum_j (g \cdot v_j)\, w_j(\delta_{ji} - w_i)
$$

$$
\boxed{\;\frac{\partial \mathcal{L}}{\partial s_i} = w_i\left[\,g \cdot v_i - \sum_j w_j\,(g \cdot v_j)\right]}
$$

A key's score is pushed **up** when its value would have helped more than the current weighted average, and **down** when it would have helped less. Attention is learned by comparison, never by supervision: nobody labels which position should attend where.

Propagating one step further, from $$s_i = \lambda\, q \cdot k_i$$:

$$
\frac{\partial \mathcal{L}}{\partial k_i} = \lambda \frac{\partial \mathcal{L}}{\partial s_i}\, q, \qquad
\frac{\partial \mathcal{L}}{\partial q} = \lambda \sum_i \frac{\partial \mathcal{L}}{\partial s_i}\, k_i
$$

### 10.3 Where the training pressure sits in this example

The target at position $$13$$ is $$\texttt{woof}$$, and $$\texttt{->}$$ alone is followed by three different tokens elsewhere in the same sequence. Any model reading only the current token pays a large loss precisely there:

$$
-\log p(\texttt{woof} \mid x_{\le 13}) \;\ge\; -\log \tfrac{1}{3} \approx 1.0986 \quad \text{for a one-token-context model}
$$

Closing that gap is the entire reason a lookup mechanism gets built.

### 10.4 Why one layer cannot learn this rule

With column-vector convention $$q_i = W_Q x_i$$, $$k_j = W_K x_j$$, the first-layer residual stream at position $$i$$ contains only token $$i$$ and its position. Therefore

$$
k_i = W_K x_i \quad\text{depends on } x_i \text{ alone},
$$

whereas the toy requires $$k_i = e(x_{i-2}) + e(x_{i-1})$$. To key on earlier context, an earlier head must first move that information into position $$i$$. Real transformers solve this with the two-layer induction circuit: a previous-token head in layer $$0$$ copies $$x_{i-1}$$ forward, and an induction head in layer $$1$$ keys on the composite. The toy collapses both layers into one hand-written line, which is exactly why it needs no depth.

### 10.5 The trained objects are products, not factors

Scores and outputs depend on the weight matrices only through two bilinear forms:

$$
q_i^\top k_j = (W_Q x_i)^\top (W_K x_j) = x_i^\top \big(W_Q^\top W_K\big) x_j
$$

$$
W_O \sum_j w_j v_j = \big(W_O W_V\big) \sum_j w_j x_j
$$

so the meaningful objects are $$W_Q^\top W_K \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$$, which decides **where** a head looks, and $$W_O W_V$$, which decides **what** it writes. Each factor is defined only up to an invertible $$M \in \mathbb{R}^{d_{\text{head}} \times d_{\text{head}}}$$:

$$
(M W_Q)^\top (M^{-\top} W_K) = W_Q^\top M^\top M^{-\top} W_K = W_Q^\top W_K
$$

$$
(W_O M^{-1})(M W_V) = W_O W_V
$$

Asking what data trains $$W_K$$ alone is therefore asking about a quantity the objective never constrains on its own.

---

## 11. Cost accounting

For one layer, one head, sequence length $$n$$, width $$d$$:

$$
\begin{array}{l|l|l}
\text{stage} & \text{shape} & \text{cost}\\ \hline
\text{projections } Q,K,V & n \times d & 3\,n\,d^2 \text{ FLOPs}\\
\text{scores } QK^\top & n \times n & n^2 d\\
\text{softmax} & n \times n & O(n^2)\\
\text{weighted sum } wV & n \times d & n^2 d\\
\text{output } W_O & n \times d & n d^2
\end{array}
$$

The toy's single query row reduces every $$n \times n$$ object to $$1 \times n$$:

$$
\text{toy, one step: } \quad 14 \text{ score operations}, \quad 14 \text{ K/V pairs projected}
$$

Generating $$T$$ tokens from a prompt of length $$n_0$$ without reuse costs

$$
\sum_{t=0}^{T-1} (n_0 + t) = T n_0 + \frac{T(T-1)}{2}
$$

which for $$n_0 = 14$$, $$T = 6$$ gives $$84 + 15 = 99$$ projections, against $$n_0 + T = 20$$ with reuse — a factor of $$4.95$$. That comparison is the opening of Part 2.

---

## 12. Summary of every expression

$$
\begin{aligned}
&\text{embed} && e(x) = \mathbf{e}_{\mathrm{id}(x)}, \quad X \in \mathbb{R}^{n \times |\mathcal{V}|}\\
&\text{key} && k_i = e(x_{i-2}) + e(x_{i-1}), \quad K = \big[(P + P^2)\bar X\big]_{0:n}\\
&\text{value} && v_i = e(x_i), \quad V = X\\
&\text{query} && q = e(x_{t-1}) + e(x_t)\\
&\text{score} && s = \lambda\, q^\top K^\top, \quad \lambda = 4\\
&\text{weights} && w_i = \mathrm{softmax}(s)_i = e^{s_i - m}\big/\textstyle\sum_j e^{s_j - m}\\
&\text{output} && \mathrm{out} = w^\top V, \quad \mathrm{out}_j = \textstyle\sum_{i : \mathrm{id}(x_i) = j} w_i\\
&\text{sample} && x_{t+1} = \arg\max_j \mathrm{out}_j\\
&\text{one line} && \mathrm{out} = \mathrm{softmax}(\lambda\, q^\top K^\top)\,V\\
&\text{real model} && \mathrm{out} = \mathrm{softmax}\big(QK^\top/\sqrt{d_{\text{head}}} + M\big)V\\
&\text{loss} && \mathcal{L} = -\textstyle\sum_t \log p(x_{t+1} \mid x_{\le t})\\
&\text{value grad} && \partial \mathcal{L}/\partial v_i = w_i g\\
&\text{score grad} && \partial \mathcal{L}/\partial s_i = w_i\big[g\cdot v_i - \textstyle\sum_j w_j (g \cdot v_j)\big]\\
&\text{circuits} && W_Q^\top W_K \ (\text{where it looks}), \quad W_O W_V \ (\text{what it writes})
\end{aligned}
$$

---

*Source: [`toy_llm_demo.py`](https://gist.github.com/chuan2019/c9b632faeb9422ca2680af9ca6066dad) — dependency-free Python, standard library only. Every number in this document was produced by running it.*
