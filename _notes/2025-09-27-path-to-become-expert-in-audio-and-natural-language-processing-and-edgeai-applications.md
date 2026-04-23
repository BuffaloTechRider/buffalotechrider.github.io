---
title: "Path to expertise: On-Device Speech & Language Processing"
date: 2025-09-27
---

# Intermediate → Expert: On-Device Speech & Language Processing

**Niche:** The person who can optimize a Conformer for ARM, build WFST decoding graphs, run a 3B LLM at 30 tok/s on an automotive SoC, and ship an end-to-end on-device voice pipeline in C++.

**Timeline:** 8–9 months at ~2 hours/day (~36 weeks)
**Philosophy:** Depth over breadth. Every concept gets implemented. Theory is learned through debugging.

---

## The Focus

**Go deep (expert-level):**
- Streaming ASR — CTC, transducers, WFST decoding, on-device Conformer
- On-device NLU — compact intent-slot models, hybrid cloud fallback
- On-device LLM inference — small LLM optimization, speculative decoding, KV-cache management, llama.cpp/MLC-LLM
- Edge model optimization — quantization, distillation, C++ inference, ARM profiling

**Go competent (working knowledge, not expert):**
- Audio signal processing — enough to debug feature pipelines
- Retrieval/ranking — enough for NLU candidate selection and on-device RAG
- LLM training/fine-tuning — enough to adapt models with LoRA, not enough to pretrain

**Deferred (separate learning tracks):**
- Learning-to-rank as a standalone discipline
- Research-level NAS, RLHF, alignment

---

## Phase 1: Audio & Transformer Foundations (Weeks 1–5)

### 1A. Speech Signal Processing (Weeks 1–2)

Competent-level. You need to understand the input to your models, not publish DSP papers.

**Sub-skills in order:**
1. Sampling theory, Nyquist, aliasing — why 16kHz for speech
2. STFT, spectrograms, mel-scale, MFCCs — the feature extraction pipeline
3. Voice Activity Detection — energy-based, then model-based
4. Noise estimation & spectral subtraction basics

**80/20 concepts:**
- MFCCs — still the backbone of on-device ASR feature extraction
- Log-mel spectrograms — the input to nearly all modern audio models
- Windowing and hop size — understand the time-frequency resolution tradeoff

**Resources:**
- *Speech and Language Processing* — Jurafsky & Martin, Ch. 16 ([web.stanford.edu/~jurafsky/slp3](https://web.stanford.edu/~jurafsky/slp3/))
- Librosa tutorials: [librosa.org/doc](https://librosa.org/doc/latest/index.html)
- Stanford CS224S lecture notes (online)

**Milestone:** Python pipeline: raw WAV → mel spectrogram → MFCC → VAD segmentation. Visualize each stage. Then port MFCC extraction to C++ using only standard math.

### 1B. Transformer & Attention Internals (Weeks 3–5)

Expert-level. This is the architecture you'll be optimizing and deploying — for both ASR and LLMs.

**Sub-skills in order:**
1. Self-attention — Q/K/V projections, scaled dot-product
2. Multi-head attention — what each head learns
3. Positional encoding — sinusoidal, learned, rotary (RoPE)
4. Layer normalization — pre-norm vs post-norm
5. KV-cache — THE bottleneck for on-device LLM inference
6. Flash Attention — memory-efficient tiling approach
7. Grouped-Query Attention (GQA) — the KV-cache compression trick used by Llama 2/3, Gemma, Phi
8. Encoder-only vs encoder-decoder vs decoder-only — when to use which

**80/20 concepts:**
- KV-cache mechanics — dominates memory on edge devices for both ASR decoders and LLMs
- GQA — reduces KV-cache size by sharing key/value heads; critical for fitting LLMs in edge memory
- Flash Attention tiling — O(n²) compute but O(n) memory
- Attention complexity — and when linear alternatives are worth the accuracy tradeoff

**Resources:**
- *Attention Is All You Need* (Vaswani et al., 2017) — implement from scratch
- Karpathy's "Let's build GPT from scratch" + nanoGPT repo
- Jay Alammar's "The Illustrated Transformer"
- Lilian Weng's "Attention? Attention!"
- GQA paper (Ainslie et al., 2023)

**Milestone:** Implement a transformer decoder from scratch in PyTorch (no nn.Transformer). Implement KV-cache with both MHA and GQA. Profile: measure memory and FLOPs for each. Understand why GQA cuts KV-cache by 4-8x.

---

## Phase 2: ASR Deep Dive (Weeks 6–12)

This is your core. Go deep.

### 2A. End-to-End ASR Architectures (Weeks 6–8)

**Sub-skills in order:**
1. CTC loss — the math, the blank token, the conditional independence assumption
2. CTC decoding — greedy, beam search, prefix beam search
3. RNN-Transducer (RNN-T) — joint network, how it differs from CTC
4. Conformer — convolution + attention, why it beats pure transformers for speech
5. Streaming ASR — chunked attention, causal convolutions, look-ahead tradeoff
6. Endpointer design — when to stop listening (energy + model-based)

**80/20 concepts:**
- CTC + beam search — how most on-device ASR works today
- Conformer architecture — current production SOTA, understand every block
- Streaming chunked attention — the latency vs accuracy knob you'll tune constantly
- RNN-T vs CTC tradeoff — RNN-T is better but heavier; know when each wins

**Resources:**
- ESPnet toolkit — reference Conformer-CTC and Conformer-Transducer implementations
- NVIDIA NeMo ASR tutorials
- Conformer paper (Gulati et al., 2020)
- RNN-T paper (Graves, 2012) + Google's streaming RNN-T papers

**Milestone:** Train a Conformer-CTC model on LibriSpeech-100h. Compare streaming (chunked) vs non-streaming WER. Profile model: FLOPs, memory, latency per utterance.

### 2B. WFST-Based Decoding (Weeks 9–10)

Expert-level. Directly relevant to your OpenFST work.

**Sub-skills in order:**
1. FST fundamentals — acceptors, transducers, weights, semirings
2. Core operations — composition, determinization, minimization, epsilon removal
3. The ASR decoding pipeline — H∘C∘L∘G graph construction
4. On-the-fly composition — when full composition is too expensive
5. Lattice generation and rescoring
6. Integrating neural model scores with WFST decoding

**80/20 concepts:**
- H∘C∘L∘G composition — understand what each transducer does and why order matters
- Determinization — why it's needed and when it blows up
- On-the-fly composition — critical for on-device where memory is constrained
- Lattice rescoring with LM — how to improve accuracy without rerunning the acoustic model

**Resources:**
- OpenFST documentation and tutorials
- Kaldi WFST recipes — the best practical WFST tutorials available
- Mohri et al., "Weighted Finite-State Transducers in Speech Recognition" (the foundational paper)
- *Speech Recognition Algorithms Using Weighted Finite-State Transducers* — Takaaki Hori & Atsushi Nakamura

**Milestone:** Build a WFST decoding graph from scratch using OpenFST: compile H, C, L, G transducers individually, compose them, decode CTC output through it. Measure WER improvement over greedy CTC decode.

### 2C. Wake Word & Keyword Spotting (Weeks 11–12)

**Sub-skills in order:**
1. Small-footprint model architectures — DS-CNN, CRNN, attention-based KWS
2. Streaming detection — sliding window, threshold tuning
3. False accept / false reject tradeoff — ROC curves for KWS
4. Data augmentation for KWS — noise injection, room impulse response simulation
5. Personalized wake words — few-shot enrollment

**80/20 concepts:**
- DS-CNN (depthwise separable CNN) — the go-to architecture for tiny keyword spotters
- Threshold tuning — the FA/FR tradeoff is the entire product decision
- Data augmentation — more important than model architecture for KWS robustness

**Resources:**
- Google Speech Commands dataset
- "Hello Edge" paper (Zhang et al., 2017) — keyword spotting on microcontrollers
- Arm ML Zoo KWS examples

**Milestone:** Train a keyword spotter (<100KB model) on Speech Commands. Deploy in C++ with streaming audio input. Tune the detection threshold to hit a target FA rate.

---

## Phase 3: On-Device NLU (Weeks 13–17)

### 3A. Intent & Slot Models (Weeks 13–16)

Expert-level. This is the brain after the ears.

**Sub-skills in order:**
1. Intent classification — multi-class, confidence calibration
2. Slot filling / NER — BIO sequence labeling
3. Joint intent-slot models — shared encoder, why joint > separate
4. Dialogue context — slot carryover, context resolution
5. Model compression for NLU — DistilBERT, TinyBERT, pruned transformers
6. On-device inference — ONNX export, INT8 quantization of NLU models
7. Hybrid routing — deciding what runs on-device vs cloud

**80/20 concepts:**
- Joint intent-slot architecture (JointBERT-style) — this is production NLU
- BIO decoding edge cases — partial entities, nested entities
- Distillation for on-device — the teacher-student pipeline for NLU
- Confidence-based routing — when to fall back to cloud (or to the on-device LLM)

**Resources:**
- *Speech and Language Processing* — Jurafsky & Martin, Ch. 8 (Sequence Labeling), Ch. 15 (Dialogue)
- Hugging Face token-classification tutorials
- Snips NLU paper — on-device NLU design reference
- MASSIVE dataset (Amazon) — multilingual NLU benchmark

**Milestone:** Build a joint intent-slot model for a car-control domain ("set temperature to 72", "navigate to home"). Train, distill to a smaller model, quantize to INT8, deploy in C++ via ONNX Runtime. Measure: accuracy, latency, model size at each step.

### 3B. Retrieval & Ranking for NLU (Week 17)

Competent-level. Supporting skill for both task-specific NLU and on-device RAG.

**Sub-skills:**
1. Bi-encoder embeddings for intent/entity retrieval
2. FAISS / HNSW for fast approximate nearest neighbor search
3. Cross-encoder reranking for disambiguation
4. On-device vector search — small index, constrained memory

**80/20 concepts:**
- Bi-encoder vs cross-encoder tradeoff — speed vs accuracy, and when to cascade
- Hard negative mining — the single biggest lever for embedding quality
- FAISS IVF-PQ — the quantized index that fits in edge memory

**Resources:**
- Manning's IR book Ch. 1-6 ([nlp.stanford.edu/IR-book](https://nlp.stanford.edu/IR-book/))
- Sentence-Transformers docs and training examples
- FAISS wiki and tutorials

**Milestone:** Build a semantic intent matcher: encode candidate intents with a bi-encoder, index with FAISS, retrieve top-K, rerank with a cross-encoder. Measure latency on a constrained device.

---

## Phase 4: Edge Optimization & Deployment (Weeks 18–25)

### 4A. Model Optimization (Weeks 18–21)

Expert-level. This is where you make models actually run on hardware. Applies to ASR, NLU, AND LLMs.

**Sub-skills in order:**
1. Post-training quantization (PTQ) — INT8, calibration strategies, per-channel vs per-tensor
2. Quantization-aware training (QAT) — fake quantization, straight-through estimator
3. INT4 and mixed-precision — which layers tolerate low precision (attention vs FFN)
4. Knowledge distillation — teacher-student, feature-based, attention transfer
5. Structured pruning — channel pruning, block pruning (actually speeds up inference)
6. Operator fusion — Conv+BN+ReLU, attention fusion, why it matters
7. Memory optimization — activation checkpointing, weight sharing, memory-mapped models

**80/20 concepts:**
- INT8 PTQ — always try this first, gets 2-4x speedup with minimal accuracy loss
- INT4 for LLM weights — the standard for on-device LLMs (GPTQ, AWQ, GGUF formats)
- Knowledge distillation — most reliable path to a small accurate model
- Structured pruning — the only pruning that speeds up inference without sparse hardware
- Operator fusion — the first thing every inference runtime does

**Resources:**
- MIT HAN Lab: TinyML and Efficient Deep Learning (Song Han) — YouTube lectures
- *Efficient Deep Learning* — Menghani (O'Reilly, 2022)
- TensorRT quantization guide
- ONNX Runtime optimization docs
- AWQ paper (Lin et al., 2023) — activation-aware weight quantization for LLMs

**Milestone:** Take your Conformer-CTC from Phase 2. Apply: INT8 PTQ → INT4 mixed-precision QAT → structured pruning → distillation into smaller architecture. Build a Pareto curve: WER vs latency vs model size vs peak memory.

### 4B. C++ Inference & Hardware Deployment (Weeks 22–25)

Expert-level. Where the rubber meets the road.

**Sub-skills in order:**
1. ONNX Runtime — export, optimize, execution providers, custom operators
2. TensorRT — engine building, layer fusion, INT8 calibration
3. ARM NEON intrinsics — SIMD for feature extraction and post-processing
4. Threading — pipeline parallelism for streaming (capture → features → inference → decode)
5. Memory management — arena allocators, avoiding fragmentation on embedded
6. Profiling — roofline model, memory bandwidth vs compute bound analysis
7. On-device model updates — differential updates, versioning

**80/20 concepts:**
- ONNX Runtime — most portable inference runtime, learn it thoroughly
- ARM NEON — where the real speedup lives for automotive/mobile targets
- Roofline model — tells you if you're memory-bound or compute-bound, determines your optimization strategy
- Pipeline parallelism — overlap audio capture with inference for streaming

**Resources:**
- ONNX Runtime docs: [onnxruntime.ai](https://onnxruntime.ai/)
- ARM NEON developer guides
- TensorRT developer guide
- Pete Warden's "TinyML" book

**Milestone:** Full C++ inference pipeline: audio capture → MFCC extraction (NEON-optimized) → Conformer inference (ONNX Runtime, INT8) → CTC+WFST decode. Run on ARM hardware (Raspberry Pi or Android NDK). Target: <200ms end-to-end for a 3-second utterance.

---

## Phase 5: On-Device LLM Inference (Weeks 26–31)

Expert-level. This is the new frontier — making language models run fast on constrained hardware.

### 5A. Small LLM Architectures & Adaptation (Weeks 26–28)

**Sub-skills in order:**
1. Small LLM landscape — Phi-3/3.5, Gemma 2B, Llama 3.2 1B/3B, Qwen2 0.5B/1.5B
2. Architecture choices at small scale — depth vs width, MoE for edge, shared embeddings
3. Tokenization for on-device — BPE, SentencePiece, vocabulary size tradeoffs (smaller vocab = smaller embedding table = less memory)
4. LoRA / QLoRA fine-tuning — adapt a base model to your domain
5. Prompt engineering for constrained models — shorter prompts = fewer KV-cache entries = faster
6. When to use an LLM vs task-specific model — the decision framework

**80/20 concepts:**
- Model selection for edge — not all 3B models are equal; GQA head count, vocab size, and context length determine memory footprint more than parameter count alone
- QLoRA — the practical way to adapt; understand that the base weights stay quantized during training
- The decision boundary — a 2MB intent classifier at 5ms beats a 2GB LLM at 500ms for "set temperature to 72". The LLM wins for "what's a good restaurant near my next meeting that has outdoor seating"
- Vocabulary size impact — 32K vocab vs 128K vocab is ~200MB difference in the embedding table at FP16

**Resources:**
- Phi-3 technical report (Microsoft, 2024)
- Llama 3 paper (Meta, 2024) — especially the efficiency sections
- Hugging Face PEFT library + QLoRA tutorials
- Sebastian Raschka's "Build a Large Language Model From Scratch"

**Milestone:** Fine-tune Llama 3.2 1B with QLoRA on your car-control domain from Phase 3. Compare accuracy, latency, and memory against your task-specific intent-slot model. Document the crossover point: at what query complexity does the LLM start winning?

### 5B. LLM Inference Optimization for Edge (Weeks 29–31)

This is the deep expertise that's rare and valuable.

**Sub-skills in order:**
1. KV-cache management — pre-allocation, rolling buffer for streaming, memory-mapped KV
2. Speculative decoding — draft model + verify, why it gives free speedup on edge
3. Continuous batching on single-user devices — interleaving prefill and decode phases
4. Weight formats — GGUF, GPTQ, AWQ, EXL2 — understand the tradeoffs (accuracy, speed, compatibility)
5. llama.cpp internals — GGML tensor library, quantization kernels, metal/NEON backends
6. MLC-LLM / ExecuTorch — alternative runtimes, when each is better than llama.cpp
7. Context length management — sliding window attention, StreamingLLM, when to truncate vs summarize
8. Prefill vs decode optimization — prefill is compute-bound, decode is memory-bound; different optimizations for each
9. On-device RAG — small vector index + LLM for grounded responses without cloud

**80/20 concepts:**
- Prefill is compute-bound, decode is memory-bound — this single insight determines your entire optimization strategy. Prefill benefits from parallelism and fusion. Decode benefits from quantization and memory bandwidth.
- Speculative decoding — use a tiny draft model (or n-gram) to propose tokens, verify in parallel with the big model. 2-3x speedup for free on edge where memory bandwidth is the bottleneck.
- KV-cache is the memory wall — a 3B model with 4K context at FP16 KV-cache uses ~1.5GB just for KV. GQA + INT8 KV quantization + rolling window brings this to manageable levels.
- llama.cpp GGUF format — the de facto standard for on-device LLM deployment. Understand Q4_K_M, Q5_K_M, Q8_0 quantization levels and their accuracy/speed tradeoffs.
- On-device RAG — retrieve from a local FAISS index, inject into a short prompt, generate with the local LLM. No cloud needed. Context window is precious — retrieval quality directly determines generation quality.

**Resources:**
- llama.cpp repo + GGML documentation (github.com/ggerganov/llama.cpp)
- MLC-LLM docs (mlc.ai/mlc-llm)
- ExecuTorch docs (pytorch.org/executorch)
- Speculative decoding paper (Leviathan et al., 2023)
- StreamingLLM paper (Xiao et al., 2023)
- "Efficient Memory Management for Large Language Model Serving with PagedAttention" (vLLM paper — the concepts apply to edge too)

**Milestone:** Deploy Llama 3.2 1B (Q4_K_M) on ARM hardware using llama.cpp. Implement:
1. Baseline: measure tokens/sec for prefill and decode separately
2. Add speculative decoding with a tiny draft model — measure speedup
3. Add INT8 KV-cache quantization — measure memory reduction vs quality
4. Build on-device RAG: FAISS index of car manual chunks + LLM generation
5. Target: 20+ tokens/sec decode on Raspberry Pi 5 or equivalent ARM SoC

---

## Phase 6: Full Integration (Weeks 32–36)

### The Complete On-Device Voice + Language Pipeline

Build the full system in C++:

1. **Wake word detection** — small CNN, always-on, <1MB
2. **Streaming ASR** — chunked Conformer-CTC with WFST decoding
3. **Intent router** — fast classifier that decides: task-specific NLU vs on-device LLM vs cloud
4. **On-device NLU (fast path)** — distilled intent-slot model, <10ms for simple commands
5. **On-device LLM (complex path)** — quantized small LLM for open-ended queries, with optional RAG
6. **Hybrid cloud fallback** — confidence-based routing when on-device can't handle it
7. **Graceful degradation** — full offline capability for core commands

**Expert concepts to integrate:**
- Three-tier routing: task-specific (fast) → on-device LLM (medium) → cloud (slow but capable)
- Memory budget management — ASR model + NLU model + LLM sharing a fixed pool; LLM weights can be memory-mapped and paged in/out
- Model hot-swapping — loading domain-specific NLU or LoRA adapters on demand
- Latency profiling of the full pipeline, not just individual models
- Endpointer → ASR → router → NLU/LLM pipeline with proper state management

**Milestone:** Demo running on a single-board computer:
- Wake-to-response <500ms for simple on-device commands (NLU fast path)
- Wake-to-first-token <1.5s for complex queries (LLM path)
- Full offline operation for core car-control commands
- Write a technical doc covering architecture decisions, tradeoffs, and profiling results

---

## Competence Track (Lighter Touch)

### Audio Beyond Speech (~1 week, during Phase 2)
- Speaker embeddings (x-vectors, ECAPA-TDNN) for personalization
- Speech enhancement via T-F masking
- **Read:** SpeechBrain tutorials for speaker verification
- **Build:** A speaker verification demo — enroll 3 speakers, verify from new audio

---

## Daily Practice Plan

**Weekdays (2 hours/day):**

| Block | Duration | Activity |
|---|---|---|
| Theory | 40 min | Read textbook/paper for current sub-skill. Handwritten or markdown notes. |
| Build | 50 min | Implement what you just read. No copy-paste. Type it, debug it, understand it. |
| Reflect | 20 min | Learning log: what clicked, what's fuzzy, one question for tomorrow. |
| Queue | 10 min | Bookmark tomorrow's resources, write down the starting point. |

**Weekend (one day, 3–4 hours):**

| Block | Duration | Activity |
|---|---|---|
| Project | 2 hours | Milestone work — the bigger builds that tie concepts together. |
| Paper | 45 min | One paper per week. Structured notes: problem, method, result, limitation. |
| Share | 15 min | Blog post, SO answer, or internal tech doc. Teaching solidifies understanding. |

---

## Progress Checkpoints

| Week | You Should Be Able To... |
|---|---|
| 5 | Explain MFCCs and transformer attention from memory. Working KV-cache with GQA implemented and profiled. |
| 12 | Train a Conformer-CTC, build an H∘C∘L∘G decoding graph, deploy a keyword spotter in C++. |
| 17 | Build and compress a joint intent-slot model. Build a semantic retrieval pipeline for NLU. |
| 25 | Quantize, prune, distill, and deploy a model in C++ on ARM. Hit latency targets. Explain roofline model. |
| 31 | Run a quantized LLM on ARM at 20+ tok/s. Implement speculative decoding. Build on-device RAG. Explain prefill vs decode optimization. |
| 36 | Run a full wake-word → ASR → router → NLU/LLM pipeline on-device. Three-tier routing working. Full offline mode for core commands. |

---

## Deferred Learning Tracks

Revisit these when your core expertise is solid (after week 36+):

### Track A: Learning to Rank
- Pointwise, pairwise, listwise losses
- LambdaMART, ColBERT, neural ranking
- Feature engineering for ranking
- Online learning to rank, click models
- **Entry point:** *Learning to Rank for Information Retrieval* — Tie-Yan Liu

### Track B: LLM Alignment & Advanced NAS
- RLHF, DPO, constitutional AI
- Hardware-aware Neural Architecture Search
- Multi-objective NAS for edge targets
- **Entry point:** Lilian Weng's RLHF blog + MIT HAN Lab NAS lectures

---

## Key Principle

**Depth beats breadth.** An expert in on-device speech + LLM inference who can ship a full voice pipeline in C++ is far more valuable than someone who's intermediate at all four fields. The hardware will tell you exactly what you don't know — deploy early, profile often, and let the constraints teach you.
