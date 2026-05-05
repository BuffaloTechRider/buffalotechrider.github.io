---
title: "Zero-Shot Confidence Estimation for Small LLMs: When Supervised Baselines Aren't Worth Training"
collection: publications
permalink: /publication/2026-05-01-Zero-shot-confidence-estimation-for-small-LLMs
date: 2026-05-01
venue: 'arXiv preprint'
category: manuscripts
citation: 'Luong N. Nguyen, &quot;Zero-Shot Confidence Estimation for Small LLMs: When Supervised Baselines Aren&apos;t Worth Training.&quot; arXiv preprint arXiv:2605.02241, 2026.'
---

[Read the paper on arXiv](https://arxiv.org/abs/2605.02241){:target="_blank"} | [Code & Data](https://github.com/BuffaloTechRider/zero-shot-llm-confidence-estimation){:target="_blank"}

## Summary

This paper asks whether you actually need labeled training data to route queries between a cheap local LLM and an expensive cloud model. The answer is no. Average token log-probability — available for free from the first query — matches supervised routing (RouteLLM-style) in-distribution and significantly outperforms it when the query distribution shifts.

We tested across 3 model families (Qwen-2.5-7B, Llama-3.1-8B, Mistral-7B), 2 datasets (MMLU-Pro and TriviaQA), ~4,500 evaluation queries, and $123 in total cloud costs. The key finding: supervised routers learn properties of the query distribution (query-side), while zero-shot signals like log-probability measure properties of the model's own generation (generation-side). When the distribution shifts, query-side signals collapse; generation-side signals transfer or improve.

We also propose retrieval-conditional self-assessment (GSA v3), a pre-generation confidence signal that achieves 3–10× lower latency than logprob by selectively injecting retrieved knowledge without revealing retrieval absence to the model.

All code, data, and the SQLite evaluation database are open-sourced for full reproducibility.
