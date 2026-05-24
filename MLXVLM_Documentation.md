# MLX-VLM: Vision-Language Models on Apple Silicon

`mlx-vlm` is a specialized Python framework built specifically to deploy and run large Vision-Language Models (VLMs) leveraging Apple's native `mlx` machine learning library. 

## 🏎️ Why MLX-VLM Matters on Mac
Standard libraries typically rely heavily on PyTorch and CUDA. When running locally on Macs, traditional execution pipelines load both the model architecture and giant tensors through CPU emulation or sluggish MPS abstraction layers, slowing everything to a crawl. 

`mlx-vlm` takes full advantage of **Apple Silicon's Unified Memory Architecture**. Because the CPU, GPU, and NPU all share the same massive memory pool on M-series chips (like your M4), `mlx-vlm` avoids the costly memory-copy overhead seen on discrete PC GPUs. This enables extremely fast, battery-efficient, local-first inference for massive vision models.

## ✨ Key Features
- **Local On-Device Generation:** Operates entirely locally; completely private and doesn't require cloud API keys.
- **Unified Memory Efficiency:** Caches and computes gradients/arrays exclusively in `mlx.core` space.
- **Multimodal Architectures:** Excellent support natively for Qwen-VL, LLaVA, IDEFICS, and Pixtral. 

---

## 💻 Standard Implementation Pattern

To leverage `mlx-vlm` in your own scripts, you follow a straightforward pattern: Load the model -> Process the prompt -> Generate output.

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

# 1. Load the model directly from the HuggingFace MLX community conversion
model_path = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
model, processor = load(model_path)

# 2. Template Generation (Critical for specific Vision Models like Qwen2-VL)
raw_prompt = "What is this image about?"
prompt = apply_chat_template(processor, model.config, raw_prompt, num_images=1)

# 3. Generate response with exact local constraints
output = generate(
    model=model,
    processor=processor,
    prompt=prompt,
    image="path/to/frame.jpg",
    max_tokens=60,
    verbose=False
)
```

---

## ⚠️ Important Developer Gotchas (The HuggingFace Fast Processor Bug)

As discovered in the `MehvishLog` build, certain implementations of `mlx-vlm` occasionally clash with the upstream HuggingFace `transformers` package.

### The Problem
When running models like `Qwen2-VL`, newer versions of `transformers` (`v4.45+`) default to using a **fast image processor** under the hood. However, this specific fast processor strictly enforces returning PyTorch (`pt`) tensors, crashing native MLX arrays with the error:
> `ValueError: Only returning PyTorch tensors is currently supported.`

### The Workaround
Because `mlx-vlm` internally calls `return_tensors="mlx"`, it triggers this strict check. The current workaround involves monkey-patching or editing `mlx_vlm/utils.py` directly in the environment's `site-packages`:

```python
# In `venv/lib/python/site-packages/mlx_vlm/utils.py` around prepare_inputs()

# Fallback gracefully to PyTorch extraction
try:
    inputs = processor.process(..., return_tensors="mlx")
except ValueError:
    inputs = processor.process(..., return_tensors="pt")
    # Detach PyTorch bindings and pass pure numpy objects back to MLX
    for k, v in inputs.items():
        if hasattr(v, "detach"):
            inputs[k] = v.detach().cpu().numpy()
```
This forces HuggingFace to do its PyTorch constraints in a sandbox while allowing Apple MLX to ingest the pure extracted pixel-numpy matrices underneath.

---
*Generated directly for the MehvishLog codebase by Antigravity.*
