# Complexity-ML

**Attention-free language modeling with fixed-state inference.**

Complexity-ML studies whether sequence transport and token-specific computation can be separated without query–key–value attention. Our current research architecture is a causal language model built from shared dilated convolutions, tied lexical objects, and narrow deterministic micro-expert residuals.

## Current architecture

```text
Token IDs
   │
   ├──► tied lexical object ───────────────┐
   │                                      │
   ▼                                      ▼
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

The earlier Mu-guided GQA and token-routed Transformer experiments remain useful controls, but they are no longer the canonical architecture.

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
| [complexity-framework](https://github.com/Complexity-ML/complexity-framework) | Training, evaluation, ablations, and model definitions |
| [vllm-cuda_graph](https://github.com/Complexity-ML/vllm-cuda_graph) | vLLM integration for fixed-state dilated-convolution inference |

The precompiled Linux/Python 3.12/CUDA 12.8 wheel is available in the [v0.3.0 release](https://github.com/Complexity-ML/vllm-cuda_graph/releases/tag/v0.3.0).

## Status

This is active research. Throughput claims are hardware- and workload-specific, and architecture claims are limited to the realized attention-free checkpoints and configurations being evaluated. Transformer/QKV variants are retained as controls, not presented as the deployment target.
