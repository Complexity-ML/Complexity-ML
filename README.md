# Complexity-ML

**Open research on fixed-state inference and deterministic lexical computation.**

Complexity-ML studies two related questions:

1. Can sequence transport and token-specific computation be separated without query–key–value attention?
2. Can token identity select a small, useful residual parameter subspace when a shared dense path already handles contextual computation?

These are separate research tracks with separate realized architectures and evidence.

## Current attention-free architecture

The current attention-free architecture is a causal language model built from shared dilated convolutions, tied lexical objects, and narrow deterministic micro-expert residuals.

```text
Token IDs
   │
   ├──► tied lexical object ───────────────┐
   │                                       │
   ▼                                       ▼
Embedding ─► shared dilated causal-convolution stack ─► lexical residuals ─► tied LM head
                  │
                  └── fixed-size decode state
```

### Shared causal-convolution substrate

Every token passes through the same stack of depthwise causal convolutions. Dilations expand the finite receptive field while keeping incremental decoding simple and deterministic.

### Tied lexical objects

A low-rank object indexed by token identity is shared across layers and tied to the embedding/output space. It provides token-specific structure without replacing the shared sequence-processing path.

### Narrow deterministic micro-experts

Small token-routed residuals add lexical specialization. Routing is deterministic and inexpensive; the shared convolutional substrate remains active for every token.

## What the canonical model does not use

The current attention-free model has:

- no query, key, or value projections;
- no attention score matrix or softmax attention;
- no growing KV cache;
- no selective scan or Mamba/SSM computation;
- no learned routing network.

This list applies only to the attention-free architecture. The matched token-routing study below uses a Transformer backbone and is reported as a separate controlled experiment.

## Token-identity residual routing study

The revised paper, **Token Identity as a Routing Signal for Residual MLP Experts**, evaluates a deliberately asymmetric Transformer MLP:

```text
Contextual hidden state x_t
   ├──► shared dense SwiGLU ──────────────────────┐
   └──► two narrow residual experts selected     ├──► feed-forward output
         by a fixed layer-specific token-ID table┘
```

Every token traverses the shared dense SwiGLU branch. Token identity selects two of four narrow residual experts, but both the shared branch and selected experts transform the same contextual hidden state. Token identity therefore controls parameter selection, not the contextual representation.

The primary experiment compares one 306.5M-parameter token-routed run with one matched dense run over 8B FineWeb-Edu tokens. At the last common checkpoint on a fixed evaluation stream drawn from the training split, the routed model reaches NLL 2.9329 and dense reaches 2.9482. The routed implementation trains more slowly. Because the comparison is single-seed and the evaluation stream is not held out, this is a matched-run observation rather than evidence of general superiority.

The final checkpoints are near parity on zero-shot ARC-Easy, PIQA and HellaSwag; the routed model reports WikiText-2 perplexity 35.20 versus 35.79 for dense.

- [Paper and reproducibility artifacts](https://github.com/Complexity-ML/tmlr-paper-pool)
- [TR-MOE-306 checkpoint](https://huggingface.co/Pacific-i64/TR-MOE-306)
- [Dense-306 checkpoint](https://huggingface.co/Pacific-i64/Dense-306)
- [Linux CPU inference runtime](https://github.com/Complexity-ML/vllm-i64)

## Fixed-state inference

Incremental decoding stores only the causal-convolution history required by each dilation. State size is independent of the number of generated tokens.

For the measured 10-layer, width-384 checkpoint:

- compact architecture-specific state: 301,056 elements, or 588 KiB per sequence in BF16;
- current vLLM uniform allocation: 1,478,400 elements, or 2.82 MiB per sequence in BF16;
- cache addresses remain stable across decode steps;
- full-sequence and incremental logits agree within numerical tolerance in unit tests.

## H100 CUDA Graph result

An official `vllm bench throughput` run on one NVIDIA H100 80GB in BF16 used 1,000 simultaneous requests, one input token and 128 generated tokens per request, with a full decode CUDA Graph captured at batch size 1,000.

| Metric | Measured value |
|---|---:|
| Elapsed time | 1.71095 s |
| Requests/s | 584.47 |
| Total tokens/s | 75,396.81 |
| Generated tokens/s | **74,812.34** |
| Generated tokens | 128,000 |

This is a saturated decode-throughput measurement, not a long-prompt prefill result. The current general-prefill path still requires kernelization before we claim competitive time-to-first-token or long-context prefill performance.

## Evidence standard

The research program uses matched controls and explicit structural tests rather than treating low language-model loss as sufficient evidence. The planned/reported evidence suite includes:

- iso-parameter GQA, convolution+dense-FFN, lexical-object, and full-model controls;
- strict-causality tests based on future-token perturbations;
- realized-model audits proving the absence of QKV parameters and attention modules;
- full-sequence versus incremental-logit equivalence;
- fixed cache-size and stable-address checks;
- associative recall, induction, and synthetic in-context-learning diagnostics across distance;
- hardware-specific prefill, decode, latency, and memory measurements with exact protocols.

## Projects

| Repository | Description |
|---|---|
| [complexity-framework](https://github.com/Complexity-ML/complexity-framework) | Training, evaluation, ablations, and realized model definitions |
| [vllm-cuda_graph](https://github.com/Complexity-ML/vllm-cuda_graph) | vLLM integration for fixed-state dilated-convolution inference |
| [tmlr-paper-pool](https://github.com/Complexity-ML/tmlr-paper-pool) | Revised token-routing manuscripts, metrics, figures, tests, and reproducibility material |
| [vllm-i64](https://github.com/Complexity-ML/vllm-i64) | Linux CPU inference for the matched TR-MOE-306 and Dense-306 checkpoints |

The precompiled Linux/Python 3.12/CUDA 12.8 wheel is available in the [v0.3.0 release](https://github.com/Complexity-ML/vllm-cuda_graph/releases/tag/v0.3.0).

## Status

This is active research. Throughput claims are hardware- and workload-specific. Architecture claims are limited to the corresponding realized checkpoints and configurations: attention-free fixed-state models and token-routed Transformer controls are reported separately.
