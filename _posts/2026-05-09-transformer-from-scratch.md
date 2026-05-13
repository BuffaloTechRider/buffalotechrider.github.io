---
title: "A transformer block from scratch"
date: 2026-05-09
permalink: /posts/2026/05/transformer-from-scratch/
tags:
  - autodidact
  - llm
  - transformer
  - learning-log
---

Many people see LLMs as a magic black box. At the architecture level, they're simpler than they look. The core of every modern LLM - GPT, Claude, Gemini, Llama — is a **transformer block**. Stack one transformer block N times (typically 12 to 80), add an embedding layer at the input and an output head at the output, and you have a working ChatGPT-style LLM. That's the whole recipe.

This is the post I wish I'd read before I started. Not "what is a transformer" - there are a hundred of those. This is the specific set of questions that blocked me, the architectural framings that finally made it click, and the handful of bugs that wasted the most time. After reading this (or better, implementing alongside it), you'll be able to answer concretely: what does it mean when someone says a model has 70B parameters, what actually defines a model's context window, and why are Q/K/V three separate things.

**TL;DR**

- LLMs feel magical but at the architecture level: it's a transformer block × N + embedding at input + output head. That's it. ~97% of parameters live in the blocks; context window is defined by attention cost, KV-cache memory, and positional encoding.
- Q/K/V names come from database retrieval, not linguistics. Attention is **learned information routing** between positions, and the linguistic interpretations came after the fact.
- Transformers beat RNNs because attention removes two bottlenecks: a compressed hidden state (now every token stays distinct), and sequential gradient flow (now any position attends to any other in one step).
- Prompts don't reconfigure the model. They populate working memory (the KV-cache), and fixed weights do something useful with it. Few-shot learning works because pretraining produced thousands of latent patterns, and the prompt selects which one to apply.
- The "modern" LLM block (Llama-style) is the 2019 GPT-2 design with six swaps: RMSNorm for LayerNorm, GQA for MHA, RoPE for absolute positions, SwiGLU for GELU MLP, optional MoE, optional sliding-window attention.
- Most of my implementation bugs were the same shape: calling an op that returns a new tensor and discarding the result (`x.view(...)` with no assignment, `masked_fill` with no assignment). Python lets you do this silently.

Code is in a standalone repo: [`transformer-block-from-scratch`](https://github.com/buffalotechrider/transformer-block-from-scratch). The README walks through the five-level exercise; if you want to try it yourself before reading the rest of this post, start there.

---

## What I Was Actually Building

The goal: a GPT-2 style decoder-only transformer block, pre-norm, with a causal mask. Five things stacked together:

```
         x  (input, shape: [B, S, d_model])
         │
         ├──────────────────────────────────┐ residual
         ▼                                  │
      LayerNorm                             │
         │                                  │
         ▼                                  │
    Multi-Head                              │
     Attention                              │
     (+ causal mask)                        │
         │                                  │
         + ─────────────────────────────────┘
         │
         ├──────────────────────────────────┐ residual
         ▼                                  │
      LayerNorm                             │
         │                                  │
         ▼                                  │
      Feed-Forward (MLP)                    │
         │                                  │
         + ─────────────────────────────────┘
         │
         ▼
      output  (shape: [B, S, d_model])
```

Two sublayers (attention, FFN), each wrapped in pre-norm + residual. Stack this N times and you've got an LLM.

What each piece does:

- **LayerNorm** keeps activations at a consistent scale. Without it, signals drift through many layers and training falls apart.
- **Multi-Head Attention** is the only place tokens communicate. It's where "this token" can look at "that token" and pull in information.
- **Feed-Forward** is per-token. No cross-token mixing. This is where most of the model's parameters live, and it's where the actual nonlinear "thinking" happens.
- **Residual connections** give gradients a direct path through the network, and let each sublayer learn a small correction rather than a full replacement.

That's the whole architecture. Everything else is detail.

---

## Making `[B, S, d_model]` Concrete

Before we go deeper, let's ground the notation. Every transformer block input is a tensor of shape `[B, S, d_model]`. If you're fuzzy on what those three letters mean geometrically, everything downstream is harder.

Imagine you're running the model on two sentences at once:

- Sentence 1: `"The cat sat"` — 3 tokens
- Sentence 2: `"A dog ran"` — 3 tokens

Then:

| Dim | Name | Value | What it represents |
|---|---|---|---|
| B | batch | 2 | Processing 2 sentences in parallel (GPU efficiency) |
| S | seq_len | 3 | Each sentence has 3 tokens |
| d_model | hidden dim | 4 here, 4096 in real LLMs | Each token is represented as a d_model-dim vector |

Input tensor shape: `[2, 3, 4]` — literally 2×3×4 = 24 numbers. Picture it as a stack of two 3×4 matrices:

```
Sentence 1 ("The cat sat"):
  Position 0 ("The"): [0.1, 0.2, 0.3, 0.4]    ← d_model=4 numbers per token
  Position 1 ("cat"): [0.5, 0.1, 0.8, 0.2]
  Position 2 ("sat"): [0.3, 0.7, 0.1, 0.9]

Sentence 2 ("A dog ran"):
  Position 0 ("A"):   [0.2, 0.1, 0.6, 0.5]
  Position 1 ("dog"): [0.9, 0.3, 0.1, 0.4]
  Position 2 ("ran"): [0.4, 0.5, 0.2, 0.8]
```

`x[1, 2, :]` is the 4-dim vector for `"ran"` (sentence index 1, position 2, all features). Those numbers didn't start as these specific values — they come from an embedding layer (a learned `[vocab_size, d_model]` lookup table that converts each token's integer ID into a d_model-dim vector).

### What "Context Window" Actually Means

The **context window** is the maximum value of `S` the model can handle. Llama 3 8B originally had 8K, extended to 128K. Gemini 1.5 Pro has 2M. Your prompt plus generated output combined must fit in `S ≤ context_window`.

What defines it in the architecture, specifically:

- **Attention is O(S²)** — doubling `S` quadruples compute. Long context is expensive.
- **KV-cache memory grows linearly with S** — every layer stores K and V for every past token. At 200K context on a 70B model, this is gigabytes per request.
- **Positional encoding determines extrapolation** — with absolute position embeddings, you literally can't extend past what was trained. With rotary embeddings (RoPE), you can go further. That's a big reason modern context windows got so large.

So "context window" isn't a single architectural property — it's the combination of training length, available inference memory, and positional scheme. Once you've built the attention yourself, you can see exactly why each of those constraints bites.

### What "Parameters" Actually Means

When you hear "Llama 3 70B has 70 billion parameters," most of those live inside the transformer block you're about to build. The breakdown for Llama 3 70B:

| Component | Params | % of total |
|---|---|---|
| Token embedding (lookup table) | ~1.05B | 1.5% |
| 80× transformer blocks | ~68B | 97% |
|   — attention projections per block | ~200M each | — |
|   — MLP per block | ~650M each | — |
| Final LayerNorm | ~8K | ~0% |
| Output head (tied to embedding) | 0 extra | 0% |

**97% of the parameters live in the stack of transformer blocks.** About two-thirds of that is in the MLPs (the feed-forward networks), one-third in the attention projections.

Once I'd built one block and counted its parameters, the scale story became concrete. Llama 3 70B isn't a fundamentally different architecture. It's 80 copies of what I just built, with `d_model=8192` and `num_heads=64`, plus some efficiency swaps.

---

## Things That Tripped Me Up

### 1. Q/K/V aren't a linguistic theory

I started with the wrong question. I thought there must be some deep reason from linguistics or cognitive science why the three matrices exist - why "queries," "keys," "values."

There isn't. Not one that came first, anyway.

**The origin is information retrieval.** The Q/K/V metaphor is database lookup: you have a query, you compare it against keys, you retrieve values. Attention is a soft, differentiable version of that - instead of exact matching, you get a weighted blend of values based on how well each key matches.

The actual history: attention came from RNN memory-access mechanisms in machine translation (Bahdanau 2014). The 2017 transformer paper's contribution was to **split the representation used for matching from the representation used for content**. That split is why you have both K and V. Using the same vector for both would couple "what I advertise" to "what I deliver," and empirically that's worse.

The linguistic interpretations came after. It turns out attention heads often learn to implement things linguists already knew (dependency relations, coreference, entity tracking), but that's emergent from training, not designed-in.

The framing that actually clicked for me: **attention is learned information routing**. Q/K/V are the parameters of a learned pub/sub system. Each token broadcasts keys/values and subscribes to a query. Through training, the model discovers which routes are useful.

Once I stopped trying to find a language-specific meaning, the mechanism was much easier to hold in my head.

### 2. Why transformers beat RNNs

I knew "transformers are better at long context." I didn't know why. The answer is two concrete bottlenecks that attention removes:

**Bottleneck 1: RNNs compress everything into a fixed-size hidden state.** To predict token 100, all information about tokens 1-99 has to fit in one vector. Early tokens get overwritten as the state updates. LSTMs push the effective range from ~10 to ~200 tokens but don't solve it.

**Bottleneck 2: RNN gradient flow is sequential.** To backprop from token 1000 to token 1, gradients multiply through 1000 sequential updates. They either vanish or explode.

Attention fixes both:

- **Tokens stay distinct.** The K and V arrays *are* the memory. When processing token 1000, you look directly at token 1's preserved representation. No compression.
- **Gradient paths are O(1).** Any token attends to any other in one step. Gradient flows back in one step too.

But then I pushed further: if the Q/K/V vectors are small (say, 128 dimensions each), how do they "remember" 200K tokens?

They don't have to. Three things do the work:

1. **The KV-cache is the actual memory.** The whole sequence's K and V are preserved and accessible. Q is just a small search query that picks from them. Like Google's index: your query is small, but it queries a billion pages.
2. **Multi-head attention runs many parallel routers per layer.** 32 heads × 96 layers = ~3000 parallel routing channels. Capacity distributed across many small heads, not crammed into one big vector.
3. **Layers integrate information progressively.** Each layer only does a little bit of mixing. Stack many and you get rich long-range integration.

That third point was the one I hadn't internalized. A single attention layer doesn't need to figure out "the whole meaning of this sequence." It just moves information around a bit, then the FFN processes it, then the next layer does it again. The depth is what makes the integration rich.

### 3. Prompts don't reconfigure the model — they populate working memory

This is the conceptual pivot that reframed how I think about prompt engineering and few-shot learning.

The model is doing one thing, always: `P(next_token | tokens_so_far)`. It's a fixed function from a prefix to a probability distribution. The prompt *is* the prefix. There's no "configuration" phase.

So why do different prompts produce different behaviors from the same weights?

**Because pretraining baked in thousands of latent patterns.** The training data contains Q&A formats, word lists, structured tables, reasoning chains, dialogue patterns. Predicting next tokens over trillions of examples forced the model to absorb all of them. At inference time, **the prompt selects which latent pattern to apply.**

Few-shot learning works through a specific mechanism: **induction heads**. These are attention heads that interpretability research has identified that implement pattern completion. Given `"sea→mer, cat→chat, apple→pomme, dog→"`, an induction head notices: "I've seen `X→Y` patterns before, and I'm at the position right after `→`, so I should attend to what follows previous `→` tokens." What follows them is French translations. The model pulls that routing decision straight from the KV-cache.

Zero weight updates. Runtime pattern matching using preserved prompt context.

The corollary that changed how I think about agent design: you're not teaching the model new things through prompts. You're building input that reliably activates the behaviors already latent in the weights. Prompt engineering is **search for the phrasing that maximizes the activation of a useful latent pattern**.

### 4. The modern block is the same thing with six swaps

The block I built is pre-norm GPT-2 architecture (2019). When I asked what frontier labs actually ship in Llama 3 or Gemini, the answer was: same skeleton, six concrete upgrades. Each solves a real problem.

| Upgrade | What it fixes |
|---|---|
| **RMSNorm** replaces LayerNorm | Faster (no mean subtraction) with no quality loss |
| **Grouped-Query Attention** replaces MHA | Cuts KV-cache memory 4-8× by sharing K/V across groups of Q heads |
| **RoPE** replaces absolute position embedding | Injects position inside attention via rotation; extrapolates better to longer contexts |
| **SwiGLU** replaces GELU MLP | Gated MLP with an extra multiplicative path; more expressive |
| **Optional: MoE** | Many expert FFNs, only a few active per token; more capability at same inference cost |
| **Optional: sliding-window attention** | Linear cost for long context at some quality loss |

It was reassuring to learn that what I built is ~95% of the modern architecture. The last 5% is efficiency engineering.