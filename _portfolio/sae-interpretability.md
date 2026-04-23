---
title: "Sparse Autoencoder — Interpretability Reference Implementation"
excerpt: "A study-and-build reimplementation of sparse autoencoders for neural network interpretability, following Anthropic's mechanistic interpretability research."
collection: portfolio
---

## Overview

A learning project and reference implementation: sparse autoencoders (SAEs) for extracting interpretable features from neural network activations. Built to internalize the mechanics behind Anthropic's [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/) work by implementing it end-to-end rather than just reading the paper.

Not novel research — a reimplementation. The value is that I've *built* the thing, so I can reason about it, not just reference it.

## What It Does

- Trains sparse autoencoders on intermediate activations from GPT-2 small
- Uses L1 sparsity penalties to learn overcomplete, monosemantic feature dictionaries
- Provides visualization tools for inspecting what each learned feature represents — including top-activating tokens and activation heatmaps
- Implements the encoder-decoder architecture described in Anthropic's research, with configurable hidden dimensions and sparsity coefficients

## Why It Matters

Understanding what neural networks are actually computing internally is one of the central challenges of AI safety. If we can decompose model activations into interpretable features, we can:

- Detect when models develop unexpected or potentially dangerous internal representations
- Understand the mechanisms behind model behaviors rather than treating them as black boxes
- Build better tools for monitoring and steering model behavior

This work is directly inspired by Anthropic's [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/) paper, which demonstrated that sparse autoencoders can extract millions of interpretable features from Claude, including features corresponding to safety-relevant concepts.

## Technical Details

- **Architecture:** Linear encoder with ReLU activation → sparse hidden layer → linear decoder
- **Loss function:** MSE reconstruction loss + L1 sparsity penalty on hidden activations
- **Training data:** Intermediate layer activations from GPT-2 small (768-dimensional)
- **Hidden dimension:** Configurable (default 4096 features from 768-dimensional input — ~5x overcomplete)

Built with PyTorch. Includes unit tests verifying forward pass shape invariants and loss non-negativity.

## Links

- **GitHub:** [BuffaloTechRider/sae-interpretability](https://github.com/BuffaloTechRider/sae-interpretability)
- **Related reading:** Anthropic's [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/) and [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/)
