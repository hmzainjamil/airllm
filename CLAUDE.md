# AirLLM — Claude Code Integration

## Purpose
Run 70B+ LLMs (Llama3, Qwen2.5, Mistral) on 4GB GPU / 8GB VRAM via layer-by-layer inference.
macOS Metal supported since v2.8.2.

## Tier 0 Routing Triggers
- "airllm" / "run 70b local" / "llm on low vram" / "layer inference" / "split model"

## Quick Use
```python
from airllm import AutoModel
model = AutoModel.from_pretrained("meta-llama/Llama-3.1-70B-Instruct")
output = model.generate(["Your prompt"], max_new_tokens=200)
```

## macOS Metal
```bash
pip install airllm[mlx]
# Uses MLX backend for Apple Silicon — fastest on M-series
```

## Best for
- Running Llama 3.1 70B locally when Ollama 7B isn't enough
- Long-context inference on local hardware (128K+ context slicing)
- Offline usage when no API keys available
