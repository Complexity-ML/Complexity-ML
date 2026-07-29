# Complexity-ML

**Open research on deterministic token routing for language models.**

Complexity-ML studies how token identity can select a small parameter subspace
while shared computation continues to process the full contextual hidden state.
TR-GQA and TR-MHA combine standard causal attention with a deterministic
Token-Routed Mixture-of-Experts feed-forward path.

## Research map

| Architecture | Sequence mixer | Feed-forward path | Current evidence |
|---|---|---|---|
| **TR-GQA** | Grouped-query attention | Shared dense SwiGLU + deterministic residual experts | Matched 306.5M / 8B-token run pair |
| **TR-MHA** | Multi-head attention | Shared dense SwiGLU + deterministic residual experts | Matched 99.5M short MPS pilot |

## TR-MoE: the shared feed-forward primitive

TR-GQA and TR-MHA use the same asymmetric feed-forward computation:

```text
token ID t ─► fixed layer-specific table ─► select two of four experts
                                              │
contextual hidden state h_t                    ▼
   ├──► shared dense SwiGLU ───────────────────┐
   └──► selected narrow residual experts ──────┴──► feed-forward output
```

Every token traverses the shared dense path. Token identity controls only which
residual parameters are selected; the shared path and the selected experts
transform the same contextual hidden state. The principal routed checkpoints
therefore use no learned routing network.

- **TR-GQA** = `attention_type="gqa"` + `mlp_type="token_routed"`
- **TR-MHA** = `attention_type="mha"` + `mlp_type="token_routed"`

The framework also contains bounded token-routed attention-adapter experiments.
They are not the architecture reported in the TR-MHA result below.

## TR-GQA result

The primary completed comparison uses one matched seed per architecture,
306.5M parameters, and an 8B-token FineWeb-Edu training budget. At the last
common evaluation checkpoint (step 7,500; 7.864B tokens processed):

| Architecture | Evaluation-stream NLL | Evaluation PPL | Training throughput |
|---|---:|---:|---:|
| Dense GQA + dense SwiGLU | 2.948246 | 19.07 | ~0.95M tok/s |
| **TR-GQA: GQA + shared top-2 TR-MoE** | **2.932897** | **18.78** | ~0.75M tok/s |

The NLL difference is -0.015349 in favor of TR-GQA at this checkpoint. This is
a token-matched, single-seed observation, not evidence of statistical
significance or general superiority. The fixed evaluation stream comes from
the FineWeb-Edu training split and is diagnostic rather than held out. The
routed implementation is also approximately 21% slower in training.

The final checkpoints are near parity on zero-shot ARC-Easy, PIQA, and
HellaSwag. The routed checkpoint reports WikiText-2 perplexity 35.20 versus
35.79 for dense. Results on additional corpora are mixed, so the observed
advantage remains modest and distribution-dependent.

- [Matched measurement table](https://github.com/Complexity-ML/tmlr-paper-pool/blob/main/supplementary_code/results/corrected_300m_scaling.csv)
- [Paper and reproducibility artifacts](https://github.com/Complexity-ML/tmlr-paper-pool)
- [TR-MOE-306 checkpoint](https://huggingface.co/Pacific-i64/TR-MOE-306)
- [Dense-306 checkpoint](https://huggingface.co/Pacific-i64/Dense-306)
- [Interactive paper companion](https://huggingface.co/spaces/Pacific-i64/Token-Routing-Interactive-Paper)

## TR-MHA pilot

A matched 99,487,680-parameter MPS pilot compares GQA, dense MHA, and
MHA + TR-MoE for 1,024,000 training tokens with seed 42:

| Architecture | Final evaluation NLL | Evaluation PPL |
|---|---:|---:|
| Dense GQA | 7.359221 | 1570.61 |
| Dense MHA | 7.369812 | 1587.34 |
| **TR-MHA: MHA + shared TR-MoE** | **7.321415** | **1512.34** |

This short, single-seed pilot validates the implementation and motivates
replication. It does not establish scaling behavior or statistical
significance. See the
[TR-MHA technical note](https://github.com/Complexity-ML/complexity-framework/blob/main/TR_MHA.md).

## Repositories

| Repository | Purpose |
|---|---|
| [complexity-framework](https://github.com/Complexity-ML/complexity-framework) | TR-GQA, TR-MHA, TR-MoE, training, evaluation, ablations, and model definitions |
| [tmlr-paper-pool](https://github.com/Complexity-ML/tmlr-paper-pool) | Token-identity routing manuscript, measurements, figures, tests, and reproducibility artifacts |
| [vllm-i64](https://github.com/Complexity-ML/vllm-i64) | Linux CPU inference for the matched TR-MOE-306 and Dense-306 checkpoints |

## Evidence standard

The research program uses matched controls and explicit structural tests rather
than treating one low language-model loss as sufficient evidence. Current
priorities include:

- multi-seed and compute-matched replication;
- independent held-out evaluation and corpus-sensitivity analysis;
- exact hardware-specific prefill, decode, latency, memory, and throughput
  protocols.

## Status

This is active research. Throughput claims are hardware- and workload-specific.
Large planned configurations are launch contracts, not completed results.
Architecture and quality claims are limited to the corresponding realized
checkpoints and reported evaluation protocols.
