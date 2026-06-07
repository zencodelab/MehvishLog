# MehvishLog

A privacy-first, local AI vision logger optimized for Apple Silicon. Uses your webcam to detect motion and generates natural-language descriptions of what it sees — entirely on-device, with zero cloud dependencies.

## How It Works

MehvishLog runs a two-stage processing loop:

1. **Motion Detection (OpenCV)** — Continuously captures webcam frames, converts to grayscale, and uses frame differencing to detect significant movement. This is lightweight and keeps CPU/battery usage minimal when nothing is happening.

2. **Vision Inference (Ollama + Gemma 4)** — When motion *is* detected and a configurable time interval has elapsed, the frame is base64-encoded and sent to the [Gemma 4 12B](https://ollama.com/library/gemma4:12b) model running locally via [Ollama](https://ollama.com). It generates a one-sentence description and logs it with a timestamp.

```
[2026-04-06 23:52:18] The image shows a person holding a box labeled "Golden" with a green label.
[2026-04-06 23:53:02] The image shows a person holding a package of wipes labeled "99% Pure Water".
```

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4/M5) — 16 GB+ unified memory recommended
- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- A webcam

## Quick Start

```bash
# 1. Install Ollama (if not already installed)
# Download from https://ollama.com or:
brew install ollama

# 2. Start the Ollama server
ollama serve

# 3. Pull the Gemma 4 12B model (~8 GB download)
ollama pull gemma4:12b

# 4. Clone the repository
git clone https://github.com/zencodelab/MehvishLog.git
cd MehvishLog

# 5. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 6. Install dependencies
pip install -r requirements.txt

# 7. Run the logger
python vision_logger.py
```

> **Note:** macOS will prompt you to grant Camera access to Terminal on the first run. Click **Allow**.

> **Note:** If you haven't pulled the model beforehand, MehvishLog will automatically pull it on first launch.

## Configuration

Tunable parameters are defined at the top of [`vision_logger.py`](vision_logger.py):

| Parameter | Default | Description |
|---|---|---|
| `MODEL_NAME` | `gemma4:12b` | Ollama model tag (swap to `gemma4:4b` for lower memory usage) |
| `INTERVAL_SECONDS` | `10` | Minimum seconds between VLM analyses |
| `MOTION_THRESHOLD` | `500` | Number of changed pixels to trigger motion |
| `DIFF_THRESHOLD` | `25` | Per-pixel intensity delta to count as changed |

## Output

Activity logs are written to `m4_vision_log.txt` in the project root:

```
[YYYY-MM-DD HH:MM:SS] One-sentence AI-generated scene description.
```

## Switching Models

You can swap the model by changing `MODEL_NAME` in `vision_logger.py`:

| Model | Size | Memory | Quality | Speed |
|---|---|---|---|---|
| `gemma4:12b` | ~8 GB | 16 GB+ | Best | ~3-5s/frame |
| `gemma4:4b` | ~2.5 GB | 8 GB+ | Good | ~1-2s/frame |

## License

[MIT](LICENSE)
