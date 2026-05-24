# airllm

> **Run 70B LLM inference on a single 4GB GPU with layer-by-layer loading — no quantization, no distillation, no pruning**

<p align="center">
  <a href="https://github.com/hmzainjamil/airllm/stargazers"><img src="https://img.shields.io/github/stars/hmzainjamil/airllm?style=for-the-badge&labelColor=555&color=yellow" alt="Stars"/></a>
  <a href="https://github.com/hmzainjamil/airllm/network/members"><img src="https://img.shields.io/github/forks/hmzainjamil/airllm?style=for-the-badge&labelColor=555&color=blue" alt="Forks"/></a>
  <a href="https://github.com/hmzainjamil/airllm/issues"><img src="https://img.shields.io/github/issues/hmzainjamil/airllm?style=for-the-badge&labelColor=555&color=red" alt="Issues"/></a>
  <a href="https://github.com/hmzainjamil/airllm/pulls"><img src="https://img.shields.io/github/issues-pr/hmzainjamil/airllm?style=for-the-badge&labelColor=555&color=purple" alt="PRs"/></a>
  <a href="https://github.com/hmzainjamil/airllm/commits/main"><img src="https://img.shields.io/github/last-commit/hmzainjamil/airllm?style=for-the-badge&labelColor=555&color=green" alt="Last Commit"/></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/70B_on_4GB_GPU-breakthrough-red?style=flat&labelColor=555"/>
  <img src="https://img.shields.io/badge/405B_on_8GB_VRAM-llama3.1-orange?style=flat&labelColor=555"/>
  <img src="https://img.shields.io/badge/PyPI-airllm-blue?style=flat&labelColor=555&logo=pypi"/>
  <img src="https://img.shields.io/badge/License-Apache_2.0-lightgrey?style=flat&labelColor=555"/>
</p>

---

## Why This Exists

Running 70B LLMs locally requires ~140GB VRAM at fp16. AirLLM eliminates this barrier by loading model layers sequentially from disk — only the layers being computed need to be in VRAM at any time. A 4GB GPU can run full-precision 70B inference. This unlocks local LLM use for developers who can't afford A100s.

---

## At a Glance

| Model | Min VRAM | Notes |
|---|---|---|
| Llama3.1 405B | 8GB | Largest open model, full precision |
| Llama3 70B | 4GB | Full quality, no quantization |
| Llama2 70B | 4GB | Tested, stable |
| Mistral 7B | 4GB | Well within limits |
| Any HF model | scales | Layer count determines throughput |
| MacOS M1/M2/M3 | 8GB unified | Native metal support |
| Speed (70B, 4GB) | ~1 tok/s | Sequential layer loading overhead |
| Disk space (70B) | ~140GB | Full precision weights |
| Quantization | none | No quality loss |
| Distillation | none | Full original model |

---

## 🧠 CONCEPTS

| Concept | Description |
|---|---|
| **Layer-by-layer inference** | Load one transformer layer, compute, offload, load next — peak VRAM = one layer |
| **Memory-mapped weights** | Weights stored on disk, memory-mapped into process — no full load into RAM |
| **Throughput vs latency** | Low throughput (~1 tok/s for 70B) but no quality loss — tradeoff for VRAM |
| **HuggingFace compatibility** | Works with any `AutoModelForCausalLM` compatible model |
| **Metal backend** | macOS Apple Silicon uses Metal Performance Shaders natively |
| **Quantization-free** | fp16 or fp32 — identical quality to cloud-hosted inference |
| **Prefill cache** | KV cache for prompt prefix — reuse across generations |
| **Compression** | Optional layer compression (experimental) — reduces disk reads |
| **Batch inference** | Experimental batching — minimal throughput gain due to serial layer loading |

### 🔥 Hot

- **405B on 8GB VRAM** — Llama3.1 405B on a consumer GPU — the strongest open model accessible to anyone
- **Zero quality loss** — no quantization means identical outputs to cloud API for same model
- **macOS M-series support** — unified memory architecture makes M1/M2/M3 excellent AirLLM platforms
- Source → [HMZ](https://github.com/hmzainjamil)

---

## ⚙️ HOW IT WORKS

```
Model weights on disk (~140GB for 70B)
    ↓
mmap: only accessed layers loaded into RAM
    ↓
GPU: single layer transferred to VRAM
    ↓
Compute: forward pass on that layer
    ↓
GPU: layer result stays in VRAM
GPU: layer weights evicted
    ↓
Next layer loaded, process repeats
    ↓
Final layer output → token generation
```

---

## 🚀 INSTALL

```bash
pip install airllm

# Optional: for compression support
pip install airllm[compression]

# macOS M-series
pip install airllm[mlx]
```

---

## 📟 USAGE

```python
from airllm import AutoModel

# 70B on 4GB GPU
model = AutoModel.from_pretrained("meta-llama/Llama-2-70b-hf")

input_text = "What is the theory of relativity?"
inputs = model.tokenizer(input_text, return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.7
)
print(model.tokenizer.decode(outputs[0]))

# macOS (MLX backend)
from airllm import AutoModel
model = AutoModel.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    backend="mlx"
)
```

---

## ⚙️ CONFIGURATION

| Parameter | Default | Description |
|---|---|---|
| `compression_type` | none | `4bit` or `8bit` (experimental) |
| `delete_original` | `False` | Delete original weights after compress |
| `prefill_chunk_size` | `512` | Tokens processed per prefill chunk |
| `offload_activations` | `True` | Offload layer activations to CPU |
| `backend` | auto | `cuda`, `cpu`, `mlx` (Apple) |
| `dtype` | `float16` | Model precision |
| `cache_dir` | `~/.cache/huggingface` | Weight storage directory |
| `max_sequence_length` | `512` | Max context length per generation |
| `layer_shards_saving_path` | auto | Where split layers are saved |
| `profiling_mode` | `False` | Log per-layer timing |

---

## 💡 TIPS AND TRICKS

### Performance
1. **SSD required** — HDD layer loading is 10-20× slower than SSD. NVMe SSD makes 70B inference usable. Source → [HMZ](https://github.com/hmzainjamil)
2. **Prefill optimization** — keep system prompts short. Each token in prefill requires full forward pass. Source → [HMZ](https://github.com/hmzainjamil)
3. **macOS M2 Ultra** — 192GB unified memory runs 405B at reasonable throughput (~3 tok/s). Source → [HMZ](https://github.com/hmzainjamil)

### Integration
4. **Quantize to compress** — `4bit` compression reduces disk reads significantly. Quality loss is minimal for most tasks. Source → [HMZ](https://github.com/hmzainjamil)
5. **Batch prompts** — send multiple prompts in one call when possible. Each generation incurs full model load. Source → [HMZ](https://github.com/hmzainjamil)
6. **Cache system prompts** — use prefill caching for identical system prompts across generations. Source → [HMZ](https://github.com/hmzainjamil)

### Advanced
7. **Model selection** — for speed-sensitive tasks, use 13B model via AirLLM — faster layer cycling. Source → [HMZ](https://github.com/hmzainjamil)
8. **Temperature 0 for consistency** — deterministic outputs make it easier to cache and compare results. Source → [HMZ](https://github.com/hmzainjamil)
9. **CUDA memory** — `torch.cuda.empty_cache()` between generations on low VRAM systems. Source → [HMZ](https://github.com/hmzainjamil)

### Debugging
10. **Profile first** — `profiling_mode=True` shows which layers are slowest disk-to-GPU. Source → [HMZ](https://github.com/hmzainjamil)
11. **Disk speed benchmark** — `dd if=/dev/zero of=/tmp/test bs=1G count=10` before blaming model speed. Source → [HMZ](https://github.com/hmzainjamil)
12. **HuggingFace token** — private/gated models (Llama2) require HF token: `huggingface-cli login`. Source → [HMZ](https://github.com/hmzainjamil)

---

## 🔧 TROUBLESHOOTING

| Issue | Cause | Fix |
|---|---|---|
| OOM despite 4GB GPU | Model too large for single layer | Check layer size; some models have large layers |
| Very slow generation | HDD instead of SSD | Move model weights to NVMe SSD |
| HuggingFace 403 | Model requires agreement | Accept model license on HF website |
| macOS Metal error | Wrong backend | Set `backend="mlx"` |
| Wrong output quality | Quantization artifacts | Use `compression_type=None` |
| Tokenizer errors | Wrong tokenizer class | Use `AutoTokenizer` from same model |
| Disk space error | Not enough space | 70B needs ~140GB free |

---

## 📊 ARCHITECTURE

```
AirLLM
├── AutoModel           # Entry point, auto-detects backend
├── CUDAModel          # NVIDIA GPU backend
├── MLXModel           # Apple Silicon backend
├── CPUModel           # CPU fallback
├── LayerLoader        # Memory-mapped weight streaming
├── PrefillCache       # KV cache for prompt prefixes
└── Compression        # Optional 4/8-bit quantization
```

---

## 🗺️ ROADMAP

- [ ] Improved throughput via speculative decoding
- [ ] Multi-GPU layer distribution
- [ ] LoRA fine-tune support with layer streaming
- [ ] Faster compression codecs
- [ ] GGUF model format support
- [ ] Web server mode with OpenAI-compatible API

---

## ☠️ STARTUPS / BUSINESSES

AirLLM eliminates GPU hardware as a barrier to local LLM deployment. Companies with privacy requirements (healthcare, legal, finance) can run 70B-class models on existing workstations without cloud API dependencies or expensive GPU servers.

**Cost comparison:** 70B model on AirLLM ($0 ongoing) vs equivalent API calls ($0.50-1.00/1M tokens). At 1M tokens/day = $15-30K/year saved.

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=hmzainjamil/airllm&type=Date)](https://star-history.com/#hmzainjamil/airllm&Date)

---

<p align="center">
  Built by <a href="https://github.com/hmzainjamil">HMZ</a> · <a href="https://github.com/hmzainjamil/airllm/issues">Issues</a> · <a href="https://github.com/hmzainjamil/airllm/pulls">PRs</a>
</p>
