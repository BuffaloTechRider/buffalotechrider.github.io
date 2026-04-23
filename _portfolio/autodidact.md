---
title: "Autodidact — A Local-First AI Agent That Learns"
excerpt: "Open-source AI agent with a local brain: thinks locally first, escalates to the cloud when uncertain, and distills what it learns back down so the lesson is permanent."
collection: portfolio
---

## Overview

Autodidact is an open-source AI agent I'm building around a simple idea: most queries don't need a frontier model. A well-equipped local model running on your machine can answer them for free, in milliseconds, without a round-trip to a cloud API. The frontier model is reserved for when the local one is out of its depth — and each time that happens, the local model learns a little more.

The guiding principle: **the search is temporary; the learning is permanent.**

## How It Works

When a query comes in, the agent routes it through three stages:

1. **Local brain first.** A local model (running on-device) takes the query, inspects its memory, and tries to answer. If it's confident, it responds immediately. Zero cloud cost, zero latency beyond local inference.
2. **Cloud fallback on uncertainty.** When the local model isn't confident, the query escalates to a more capable cloud model.
3. **Learning loop.** The response from the cloud isn't just returned to the user — it's distilled into the local memory so that next time a similar query arrives, the local brain handles it directly.

The effect: the more the agent is used, the less often it needs the cloud. The operating cost curve bends toward zero as the local brain grows.

## The Interesting Problems

Building this has surfaced a set of problems I didn't see coming when I started:

- **Confidence calibration.** "Route to the cloud when the local model isn't confident" sounds simple. In practice, self-reported model confidence (token log-probabilities, verbal hedging) is famously poorly calibrated — a model can be confidently wrong, or unnecessarily hedging when it actually knows the answer. Getting this right is an open research problem, not a solved one.
- **What exactly gets "learned."** The distillation step is deceptively deep. Do you store raw text in a retrieval cache? Do you fine-tune the local model? Do you train a small LoRA adapter? Each has different tradeoffs around latency, catastrophic forgetting, and whether the "knowledge" actually generalizes or is just memorized.
- **Cost / latency / quality as a live tradeoff.** Once you have a routing layer, you have a three-axis optimization. Most LLM products treat this implicitly. Making it explicit is where the engineering lives.

## Status

Work in progress. Open-sourcing as I go. This page will be updated with the GitHub link, architecture diagram, and a technical blog post once the first working version lands.

## Why This Matters

The shape of the problem — routing between local and cloud models, deciding when a frontier capability is warranted, capturing what was learned — parallels how safety-critical AI products have to think about graceful degradation and capability/cost tradeoffs. It's the same discipline I've been practicing shipping automotive AI, applied to a different surface.

## Links

- GitHub: coming soon
- Related reading: Anthropic's work on [model routing and cost-aware deployment](https://www.anthropic.com/research), [distillation research](https://arxiv.org/abs/1503.02531)
