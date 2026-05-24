# MehvishLog

A privacy-first, local AI vision logger optimized for Apple Silicon. Uses your webcam to detect motion and generates natural-language descriptions of what it sees — entirely on-device, with zero cloud dependencies.

## How It Works

MehvishLog runs a two-stage processing loop:

1. **Motion Detection (OpenCV)** — Continuously captures webcam frames, converts to grayscale, and uses frame differencing to detect significant movement. This is lightweight and keeps CPU/battery usage minimal when nothing is happening.

2. **Vision Inference (MLX-VLM)** — When motion *is* detected and a configurable time interval has elapsed, the frame is analyzed by the [Qwen2-VL-2B-Instruct-4bit](https://huggingface.co/mlx-community/Qwen2-VL-2B-Instruct-4bit) model running natively on Apple's Unified Memory via `mlx-vlm`. It generates a one-sentence description and logs it with a timestamp.

```
[2026-04-06 23:52:18] The image shows a person holding a box labeled "Golden" with a green label.
[2026-04-06 23:53:02] The image shows a person holding a package of wipes labeled "99% Pure Water".
```

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- A webcam

## Quick Start

```bash
# Clone the repository
git clone https://github.com/zencodelab/MehvishLog.git
cd MehvishLog

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the logger
python vision_logger.py
```

> **Note:** The first launch downloads ~1.5 GB of model weights from HuggingFace. Subsequent runs use the local cache and start instantly.

> **Note:** macOS will prompt you to grant Camera access to Terminal on the first run. Click **Allow**.

## Configuration

Tunable parameters are defined at the top of [`vision_logger.py`](vision_logger.py):

| Parameter | Default | Description |
|---|---|---|
| `INTERVAL_SECONDS` | `5` | Minimum seconds between VLM analyses |
| `MOTION_THRESHOLD` | `500` | Number of changed pixels to trigger motion |
| `DIFF_THRESHOLD` | `25` | Per-pixel intensity delta to count as changed |

## Output

Activity logs are written to `m4_vision_log.txt` in the project root:

```
[YYYY-MM-DD HH:MM:SS] One-sentence AI-generated scene description.
```

## Documentation

- [MLXVLM_Documentation.md](MLXVLM_Documentation.md) — MLX-VLM reference guide and known HuggingFace gotchas

## License

[MIT](LICENSE)
