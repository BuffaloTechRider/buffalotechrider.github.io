---
title: "Things I got wrong building a confidence evaluator for local LLMs"
date: 2026-04-25
permalink: /posts/2026/04/confidence-evaluator-for-local-llm/
tags:
  - autodidact
  - llm
  - calibration
  - routing
  - applied-ai
---

*An in-progress lab-notebook post from the Autodidact project.*

Injecting retrieved context into a "should I answer?" confidence gate made the signal *worse*, not better. That was the first of four surprises I didn't see coming while building a confidence evaluator for local LLMs.

**TL;DR**

- RAG injected into "should I answer" gates can lower the signal by 0.08 AUROC, not raise it. In some cases, the model has the answer in its weights but the retrieval makes it second-guess itself.
- Self-assessment prompts vary wildly by model. Mistral-7B's "direct" prompt hits 0.747 AUROC. Llama 3.1 at 8B is barely above chance with any prompt. Pick your model for calibration, not just capability.
- Naive 6-signal ensembles can underperform the best single signal. Thompson Sampling fusion at 0.564 AUROC lost to a single logprob signal at 0.642.
- Retrieval quality is often the real bottleneck. If only 10/60 of your eval queries return any retrieval hit, you're measuring noise, not RAG.

Numbers from real runs on MMLU-Pro against Ollama + AWS Bedrock. None of this is final - the v0.1 main experiment is still running.

---

I've been building **Autodidact**, a local-first AI agent framework. The central piece is a **confidence evaluator** - something that decides whether a small local model (Qwen 2.5 7B, Llama 3.1 8B, Mistral 7B) can answer a question, or whether to escalate to a cloud model (Claude 4.5, GPT, etc.).

Autodidact is still a project in development. I'll link the repo once v0.1 is stable enough for external eyes — until then, this post is the current state of the experiments.

If it works, you get cheap local inference most of the time and cloud only when needed. If it doesn't, you either hallucinate (local answers wrong questions) or escalate everything (pay cloud prices for no benefit).

This post is the findings I wish I'd known before starting - written while the v0.1 main experiment is still running, so treat it as a lab notebook, not a final report.

## The setup in one paragraph

Query comes in → compute six "confidence signals" (logprob uncertainty, self-consistency across two samples, knowledge-similarity, a lightweight query classifier, a learned "energy" score on query embeddings, and a grounded self-assessment where the model is asked YES/NO whether it can answer). Fuse them via Thompson Sampling. If the fused score is above a threshold, answer local; otherwise escalate to cloud and store the answer for next time. Some signals are from existing works — [Kadavath et al.](https://arxiv.org/abs/2207.05221), [Wang et al. self-consistency](https://arxiv.org/abs/2203.11171), and [RouteLLM](https://arxiv.org/abs/2406.18665). What I'm doing is composing multiple signals and measuring what actually works.

Here's what I got wrong.

## 1. "Grounding" self-assessment on retrieval actively hurt it

I had a signal called **Grounded Self-Assessment** (GSA): before answering, show the model what its knowledge store retrieved for this query, then ask "do you have enough information to answer correctly?" The intuition: when retrieved context is good, the model should be able to say Yes, I can answer this question, regardless of whether its "intelligence" alone could answer the question or not. And when it's missing, the model has to rely on its own weights to answer.

Sounds right, right? But the experiment results surprised me.

I ran an ablation with four prompt variants on qwen2.5:7b, with vs. without retrieved hits injected. The "with hits" condition had AUROC **0.538**. The "without hits" condition had AUROC **0.620**. Injecting retrieved context **lowered** the signal by 0.08 AUROC.

Why? I looked at the 5 queries where retrieval returned hits above the 0.75 similarity threshold. On two queries where the model would have answered CORRECTLY without retrieval, injecting retrieved content caused the model to say "NO, I don't have enough information." The retrieval was tangentially relevant — it didn't contain the specific answer — and the model interpreted its presence as a cue to look for the answer in the retrieval, didn't find it, and said NO.

The model had the answer in its weights. The retrieval made it second-guess itself.

**Lesson:** Retrieval-augmented confidence isn't free. "Does this retrieved content help?" is different from "Do I know the answer?" If your prompt emphasizes the former, you're measuring retrieval quality, not model knowledge. For my use case, I am still doing more experiments to make sure if dropping retrieval injection from the self-assessment prompt (keeping it only in the main-answer prompt, separately) is the right call.

This has implications beyond my project. Many RAG systems inject retrieved context into "should I answer" gates. That might be hurting them.

## 2. The prompt has significant impact on self-assessment quality and varies by model family

After fixing GSA, I ran the prompt ablation on three models: qwen2.5:7b, llama3.1:8b, mistral:7b-instruct. Four prompt variants each, 100 queries each, no retrieval injection:

1. `current`: "Do you have enough information to answer this question correctly?"
2. `direct`: "Can you answer this question correctly?"
3. `confidence`: "Are you confident you can answer this question?"
4. `prediction`: "Will your answer to this question be correct?"

Here is the result:

| Model | "current" | "direct" | "confidence" | "prediction" |
|---|---|---|---|---|
| qwen2.5:7b | 0.470 | 0.591 | **0.636** | 0.525 |
| llama3.1:8b | **0.545** | 0.535 | 0.519 | 0.515 |
| mistral:7b-instruct | 0.562 | **0.747** | 0.684 | 0.657 |

Bolded is the winner per row. Three different winning prompts across three models. And Llama's "winner" at 0.545 is barely above chance; all four variants fall inside the bootstrap CI of each other. Llama has essentially **no self-assessment signal at all** at this model size with any of these prompts.

Mistral is the most interesting. AUROC 0.747 with the "direct" prompt — the highest single-signal AUROC I've measured. Same prompt gives 0.591 on Qwen and 0.535 on Llama. The model matters, and it matters a lot.

**Lesson:** If we're going to use a self-assessment signal for the confidence evaluator, a one-size-fits-all prompt probably won't work. A mechanism to pick or adapt the prompt based on the model's family (and its post-training regime) looks more promising.

This is also consistent with the [Kadavath et al.](https://arxiv.org/abs/2207.05221) observation that calibrated self-assessment depends heavily on the RLHF program. Qwen's post-training includes self-critique data. Mistral's does something else. Llama 3.1 seems to not calibrate its confidence at all - it's been trained for helpfulness and harmlessness, not for honest uncertainty.

If you care about LLM calibration, **pick your base model for calibration, not just capability.** Mistral-7B-Instruct punches above its weight here.

## 3. Retrieval quality is my actual bottleneck, not the LLM

I ran an answer-quality study: 60 queries × 3 models × 2 conditions (with retrieval injected into the answer prompt vs. without). The delta was:

| Model | Delta (with − without) | p-value |
|---|---|---|
| qwen2.5:7b | +0.017 | 1.000 |
| llama3.1:8b | +0.033 | 0.625 |
| mistral:7b-instruct | -0.017 | 1.000 |

All statistically a wash at n=60. My first interpretation: "retrieval injection doesn't help answer quality."

That's what the data says. But look at `retrieval_recall_at_5`: **10 out of 60 queries actually had any retrieval hit** above the 0.75 similarity threshold. Across ALL three models. Same number because retrieval depends on the embedding model, not the LLM.

I'm measuring the retrieval-helps-answer effect on n=10 samples. That's not underpowered; that's uninterpretable.

The knowledge store has 500 entries. MMLU-Pro has ~14 subject categories. Most eval queries have zero category-matched entries in the store. The embedder is nomic-embed-text. Cross-encoder reranking: none. Question-to-answer rewriting (HyDE): none. It's an entry-level RAG.

**Lesson:** Before concluding "retrieval doesn't help," check whether retrieval is even happening. When it's not, you're measuring the noise floor of your LLM's hallucination rate.

The next step in the project is upgrading the retrieval pipeline (bge-large-en-v1.5 embedder + bge-reranker-base cross-encoder) and re-measuring. I have three specific predictions I'll report back on: retrieval_recall_at_5 should rise from 17% to ≥40%, knowledge-similarity AUROC should rise from 0.484 to ≥0.60, and the retrieval-injection answer-quality delta should finally swing positive.

If those predictions all hold, the confidence evaluator story gets much stronger. If they don't, I need to rethink.

## 4. Multi-signal ensembles are not automatically better

I put a lot of design effort into Thompson Sampling fusion — the Bayesian bandit that combines all six confidence signals into one routing decision. Surely the ensemble beats any single signal.

Not so much. On qwen2.5:7b:

- Single signal (logprob_uncertainty): AUROC 0.642
- Six-signal mean ensemble: AUROC 0.615
- Six-signal Thompson fusion: AUROC 0.564

The ensemble is WORSE than the best single signal, because half of my signals are near-chance (query classification keyword-based, weakly correlated with correctness) and they dilute the good signals.

This shouldn't have surprised me. It's a classic "averaging a good signal with noise gives you a worse signal" trap. Thompson Sampling eventually learns to downweight the bad signals, but the bootstrap period is long, and on a 1000-query benchmark the bandit hasn't converged yet.

**Lesson:** Simple baselines matter. Before adding ensemble fusion, measure whether one signal alone already wins. If it does, the ensemble's job is to NOT make things worse — a much weaker claim than "the ensemble is better."

The v0.1 main experiment will retest this on three models with proper statistical power. I'm not confident Thompson fusion beats single-signal, and the paper I want to write has to contend with that.

## What I still don't know

Honest list:

1. **Does logprob_uncertainty generalize cross-model?** Measured on Qwen only so far. Theory says yes (calibration theory is model-agnostic); data hasn't confirmed.
2. **Does retrieval quality actually matter for the confidence story, or is the embedder I use a red herring?** The answer is the v0.1 main experiment's gate.
3. **Is Thompson Sampling the right fusion strategy for signals that might be correlated?** Classical TS assumes independent arms. My signals are almost certainly correlated (good self-assessment and good logprobs probably co-occur). The math gets messier here than the paper implies.

## What's next

Three-week plan: finish the retrieval upgrade, re-measure the three predictions above, then run the main experiment across all 3 models at n=1000 with RouteLLM-style baselines. That produces the data for a technical report on GitHub. If it works, v0.2 runs the same experiment with growing memory (the thing I actually want to study) and probably becomes a workshop or short paper submission.

If you've built something adjacent (LLM routing, confidence estimation, RAG for local models), I'd genuinely like to hear where your findings agree or disagree with mine. Reach out at [paulnng@icloud.com](mailto:paulnng@icloud.com). The next post will land with real cross-model numbers when the main experiment finishes.

---

*Autodidact is under active construction. Nothing here is advice. It's a lab notebook made public so people can push back on it.*
