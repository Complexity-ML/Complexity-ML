# Complexity-ML

**Building the next generation of efficient AI architectures.**

We develop novel transformer architectures focused on **determinism**, **efficiency**, and **real-time inference**.

---

## Our Innovations

### INL Dynamics (Clamped)

A control system inspired by robotics, integrated into transformer layers:

```python
error = h - mu                      # deviation from equilibrium
v_next = alpha * v - beta * error   # velocity update (momentum + correction)
h_next = h + dt * gate * v_next     # position update (integration)
```

**Key features:**
- Smooth token trajectories (no jerky generation)
- PID-like stability with learnable dynamics
- Clamped parameters (`beta_max=2.0`, `velocity_max=10.0`) for training stability

---

### Token-Routed MLP (Deterministic MoE)

A radically simple approach to Mixture of Experts:

```python
expert_id = token_id % num_experts
```

| Aspect | Learned MoE | Token-Routed (Ours) |
|--------|-------------|---------------------|
| Load Balancing | Learned | **Perfect by design** |
| Routing Params | Millions | **Zero** |
| Deterministic | No | **Yes** |
| Routing Latency | 5-10ms | **<0.1ms** |

**Why it works:**
- Uniform distribution across experts
- No expert collapse
- 100% reproducible inference
- One line of code

---

## Projects

| Repository | Description | Status |
|------------|-------------|--------|
| [complexity-deep](https://github.com/Complexity-ML/complexity-framework) | Novel transformer with INL + Token-Routed MLP | Active |
| [complexity-tokenizer](https://github.com/Complexity-ML/complexity-tokenizer) | Fast BPE tokenizer in Rust with INL-BPE training | PyPI |
| [pacific-prime](https://huggingface.co/Pacific-Prime/pacific-prime) | 150M parameter model checkpoint | Training |

---

## Architecture

```
Input
  │
  ▼
[RMSNorm] → [GQA Attention] → [INL Dynamics] → [RMSNorm] → [Token-Routed MLP]
  │                                                              │
  └────────────────────── Residual ──────────────────────────────┘
  │
  ▼
Output
```

---

## Quick Start

```bash
pip install complexity-deep
pip install complexity-tokenizer
```

```python
from complexity_deep import DeepForCausalLM
from complexity_tokenizer import Tokenizer

model = DeepForCausalLM.from_pretrained("Pacific-Prime/pacific-prime")
tokenizer = Tokenizer.from_pretrained("Pacific-Prime/pacific-prime")

output = model.generate(tokenizer.encode("Hello"), max_new_tokens=50)
print(tokenizer.decode(output))
```

---

## Philosophy

> **Simplicity over complexity.** The best ideas are often the simplest.

- No learned routing when modulo works
- No complex scheduling when clamping stabilizes
- No over-engineering when one line suffices

---

## Links

- [HuggingFace](https://huggingface.co/Pacific-Prime)
- [PyPI - complexity-tokenizer](https://pypi.org/project/complexity-tokenizer/)

---

<p align="center">
  <i>Deterministic AI for a predictable future.</i>
</p>
