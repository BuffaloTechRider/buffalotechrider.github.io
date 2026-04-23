---
title: "On-Device Voice AI Pipeline"
excerpt: "End-to-end on-device speech inference pipeline with Conformer ASR and WFST decoding, engineered for strict latency and memory budgets on embedded automotive hardware."
collection: portfolio
---

## Overview

An end-to-end on-device speech recognition pipeline that runs entirely on embedded hardware — no cloud round-trip required for core voice interactions. Built for a production voice assistant deployed in safety-critical environments where network connectivity is unreliable and user-perceived latency must stay under a few hundred milliseconds.

The core challenge: run a modern neural ASR stack — wake word detection, streaming acoustic modeling, weighted finite-state transducer (WFST) decoding, and NLU — on a device with a fraction of the memory and compute of a datacenter server, while meeting hard real-time guarantees.

## What It Does

- **Wake word detection** — Always-on low-power keyword spotter that gates the rest of the pipeline. Optimized to run on a small DSP or microcontroller-class core with minimal duty cycle.
- **Audio capture and front-end** — Microphone array processing (beamforming, echo cancellation, noise suppression) feeding into MFCC and mel-spectrogram feature extraction.
- **Streaming Conformer-CTC ASR** — A Conformer-based acoustic model runs in streaming mode, emitting token posteriors chunk-by-chunk with bounded lookahead so partial transcripts are available before the user finishes speaking.
- **WFST decoding** — A pre-composed H∘C∘L∘G decoding graph (HMM topology ∘ context-dependency ∘ lexicon ∘ grammar/LM) produces the final word sequence. The composed graph is optimized offline and loaded as a memory-mapped artifact at runtime.
- **On-device NLU** — Intent classification and slot filling run locally, so common commands (media, navigation, climate) complete without ever touching the cloud.
- **Hybrid cloud fallback** — When the on-device stack's confidence is low or the query is out of domain, the request transparently escalates to a cloud endpoint.

## Why It Matters

Voice interfaces that depend on the cloud are brittle in exactly the environments where they matter most — moving vehicles, poor connectivity, privacy-sensitive contexts. Pushing inference to the edge turns voice from a "nice when it works" feature into a reliable one.

The interesting part from a systems perspective: the same engineering pressures that shape on-device inference — memory bandwidth, operator fusion, quantization error, tail latency, graceful degradation — reappear at datacenter scale when serving large language models. An 80M-parameter Conformer on a car SoC and a 100B-parameter transformer on a GPU fleet are both, fundamentally, latency- and memory-bound inference problems. The techniques transfer; only the constants change.

This work also reinforced a lesson that applies directly to frontier AI deployment: **reliability constraints are first-class design inputs, not something you bolt on at the end**. A voice assistant that fails unpredictably in a moving vehicle isn't a software bug — it's a safety issue. The same mindset applies to AI systems whose failures have real-world consequences.

## Technical Details

**Acoustic model:** Streaming Conformer-CTC, roughly 80M parameters, INT8 post-training quantized. Chunk-based streaming with limited right-context lookahead to cap emission latency.

**Decoder:** OpenFST-based WFST decoder operating on a pre-composed H∘C∘L∘G graph. The composed graph is determinized, minimized, and weight-pushed offline so the on-device decoder only performs token-passing search over an immutable, memory-mapped artifact. A grammar FST narrows the search space for in-domain commands; a larger n-gram LM backs off for open-domain queries.

**Edge optimization techniques:**
- **INT8 quantization** — Post-training quantization of Conformer weights and activations, with per-channel scales for convolution and attention projections. Calibration on a held-out audio set keeps WER regression minimal.
- **ARM NEON SIMD kernels** — Hand-tuned inner loops for matmul and depthwise convolution targeting Cortex-A class cores. Where NEON wasn't enough, the hot paths dropped into assembly-level intrinsics.
- **Operator fusion** — Layer-norm + linear + activation fused into single kernels to cut memory round-trips. Attention computed in a fused QKV projection where the model topology allowed.
- **Memory arena allocators** — All per-utterance tensors allocated from pre-sized arenas so inference runs with zero heap allocation after warm-up. This bounded memory usage and eliminated a whole class of tail-latency spikes from allocator contention.
- **Thread and core pinning** — Audio, front-end, acoustic model, and decoder pinned to specific cores to avoid scheduler jitter on a shared SoC.

**Reliability and safety:**
- **Bounded latency guarantees** — End-to-end budget tracked per stage. If a stage runs long, the pipeline degrades gracefully (e.g., shorter lookahead, smaller beam) rather than missing the deadline outright.
- **Graceful degradation** — When resources are constrained (thermal throttling, concurrent workload), the pipeline transparently reduces beam width or falls back to a smaller acoustic model rather than failing.
- **Hybrid cloud fallback** — A confidence-gated handoff to the cloud for out-of-domain queries, so users don't notice the on-device vs. cloud split.
- **Observability** — Per-stage latency histograms and decoder statistics exported for offline analysis, so regressions in tail latency or WER surface before they ship.

## Why This Matters for Large-Scale Inference

The principles driving on-device voice AI translate directly to serving frontier models at datacenter scale:

- **Quantization and operator fusion** are the same levers used for efficient LLM serving — INT8/FP8 weights, fused attention kernels, paged KV-cache layouts.
- **Memory arena allocation** mirrors the static allocation patterns used in high-throughput LLM serving to avoid allocator jitter.
- **Bounded latency and graceful degradation** are core to production LLM APIs — when load spikes, you want the system to shed work predictably, not fail unpredictably.
- **Confidence-gated fallback** is structurally similar to model routing in production LLM stacks, where small fast models handle easy queries and larger ones handle the hard cases.

Efficient inference is efficient inference, whether the model is 80M parameters on a car or 100B+ on a GPU cluster.

## Links

- **Related reading:** [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100), [OpenFST](https://www.openfst.org/), [Streaming End-to-End Speech Recognition](https://arxiv.org/abs/1811.06621)

