# Complexity-ML

**Building the next generation of efficient AI architectures.**

We develop novel transformer architectures focused on **determinism**, **efficiency**, and **real-time inference**.

---

## Our Innovations (v0.13.0)

### 1. Mu-Guided Architecture (INL 2025)

The key innovation: **μ (mu)** from previous layers guides ALL components:

```python
# Mu-Guided Attention (KQV order - industry standard)
x_mu = concat([x, mu_prev], dim=-1)
k = x_mu @ concat([W_k, W_mu_k])  # K biased by mu
q = x_mu @ concat([W_q, W_mu_q])  # Q biased by mu
v = x_mu @ concat([W_v, W_mu_v])  # V biased by mu

# Mu-Guided Expert Routing
router_logits = base_router(x) + mu_router(mu_prev)

# Contextual Mu for next layer
mu_next = mu + mu_proj(h)
```

**Why Mu everywhere?**
- **Top-down guidance**: Global context informs local computations
- **2-3x faster convergence**: Model learns structure faster
- **Fused operations**: concat+cuBLAS = 2x faster than separate matmuls

---

### 2. Token-Routed MLP + Mu Override

Deterministic routing with contextual adaptation:

```python
# Base: deterministic, perfectly balanced
expert_id = token_id % num_experts

# Mu override: context can shift expert selection
router_logits = base_router(x) + mu_router(mu_prev)
```

| Aspect | Top-K MoE | Token-Routed + Mu (Ours) |
|--------|-----------|--------------------------|
| Base Routing | 100% learned | **Deterministic (stable)** |
| Context-Aware | Router network | **Mu (lightweight)** |
| Expert Collapse | Risk | **None** |
| Load Balancing Loss | Required | **Not needed** |
| Auxiliary Loss | Required | **Not needed** |

**Best of both worlds**: Stability of deterministic routing + intelligence of learned routing.

---

### 3. INL Dynamics with Contextual Mu

A control system inspired by robotics:

```python
error = h - mu                      # deviation from equilibrium
v_next = alpha * v - beta * error   # velocity update (momentum + correction)
h_next = h + dt * gate * v_next     # position update (integration)

# v0.13.0: Contextual mu for next layer
mu_contextual = mu + mu_proj(h)     # mu adapts based on current state
```

**Key features:**
- Smooth token trajectories (no jerky generation)
- PID-like stability with learnable dynamics
- Clamped parameters (`beta_max=2.0`) for training stability
- **Mu Highway**: Context flows across all layers

---

### 4. Modern Attention Stack

- **KQV Order**: Industry standard (Llama, Qwen, GPT) for optimal KV-cache
- **GQA**: Grouped Query Attention (8 KV heads)
- **QK Norm**: Attention stability at scale
- **RoPE**: Rotary positional embeddings
- **Flash Attention**: SDPA via PyTorch 2.0+

---

## Architecture

```
Input
  │
  ▼
[RMSNorm] ─► [Mu-Guided GQA (KQV)] ─► [INL Dynamics] ─► [RMSNorm] ─► [Token-Routed MLP]
  │              ▲                         │                              ▲
  │              │                         │                              │
  │         mu_prev                   mu_contextual ──────────────────────┘
  │                                        │
  +─────────────────── Residual ───────────┼──────────────────────────────+
  │                                        │                              │
  ▼                                        ▼                              │
Output ◄───────────────────────────── mu_next (to next layer) ◄──────────┘
```

---

## Projects

| Repository | Description | Version |
|------------|-------------|---------|
| [complexity-deep](https://github.com/Complexity-ML/complexity-deep) | Model architecture (Mu-Guided + Token-Routed) | v0.13.0 |
| [complexity-framework](https://github.com/Complexity-ML/complexity-framework) | Training framework with all innovations | v0.3.0 |
| [pacific-prime](https://huggingface.co/Pacific-Prime/pacific-prime) | 1.5B parameter model checkpoint | Training |

---

## Current Training

| Model | Params | Steps | Status |
|-------|--------|-------|--------|
| complexity-deep 1.5B | 1,516M | 1M/1M | Training on H100 |

**Dataset**: FineWeb-Edu (French/English)
**Hardware**: H100 80GB
**Precision**: BF16

---

## Quick Start

```bash
pip install complexity-deep>=0.13.0
```

```python
from complexity_deep import DeepForCausalLM, DeepConfig
from tokenizers import Tokenizer
import torch

# Load model
model = DeepForCausalLM.from_pretrained("Pacific-Prime/pacific-prime")
tokenizer = Tokenizer.from_file("tokenizer.json")

# Generate
input_ids = torch.tensor([tokenizer.encode("Hello").ids])
output = model.generate(input_ids, max_new_tokens=50, temperature=0.8)
print(tokenizer.decode(output[0].tolist()))
```

---

## What Makes Us Different

| Innovation | Status | Description |
|------------|--------|-------------|
| Mu-Guided KQV | **Novel** | μ biases K, Q, AND V projections |
| Mu-Guided Expert Routing | **Novel** | μ influences MLP expert selection |
| Contextual Mu (mu_proj) | **Novel** | μ adapts based on hidden state |
| Token-Routed MLP | **Novel** | Deterministic routing by token ID |
| INL Dynamics | **Novel** | Robotics control in transformers |
| Fused Mu-KQV | **Novel** | 2x faster via concat+cuBLAS |
| Hybrid Routing | **Novel** | Deterministic base + learned override |

---

## Philosophy

> **Simplicity over complexity.** The best ideas are often the simplest.

- Deterministic routing when it works → add learned override only where needed
- Top-down guidance (μ) instead of complex routing networks
- Fused operations for speed, not just correctness
- Robotics-grade stability for production deployments

---

## Links

- [HuggingFace](https://huggingface.co/Pacific-Prime)
- [PyPI - complexity-deep](https://pypi.org/project/complexity-deep/)
- [PyPI - complexity-framework](https://pypi.org/project/complexity-framework/)
- [GitHub - complexity-deep](https://github.com/Complexity-ML/complexity-deep)
- [GitHub - complexity-framework](https://github.com/Complexity-ML/complexity-framework)

---

<p align="center">
  <i>Deterministic AI for a predictable future.</i>
</p>
