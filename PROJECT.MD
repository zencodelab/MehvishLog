# MehvishLog — Project Overview

A **privacy-first, local AI vision logger** for Apple Silicon Macs. It watches your webcam, detects motion, and uses an on-device vision-language model to describe what it sees in natural language — all offline, with zero cloud dependencies.

## Architecture

The project is a single-script Python app ([vision_logger.py](vision_logger.py)) with a **two-stage processing loop**:

1. **Motion Detection (OpenCV)** — Continuously grabs webcam frames, converts to grayscale, applies Gaussian blur, and uses frame differencing (`cv2.absdiff`) to detect significant pixel changes. This is cheap and keeps CPU/battery idle when nothing is happening.

2. **Vision Inference (Ollama + Gemma 4 12B)** — When motion is detected *and* a cooldown interval (`INTERVAL_SECONDS`, default 10s) has elapsed, the frame is base64-encoded in memory and sent to the [Gemma 4 12B](https://ollama.com/library/gemma4:12b) model via the Ollama Python client. It generates a one-sentence scene description and appends it with a timestamp to `m4_vision_log.txt`.

## Key Configuration

Tunable parameters at the top of [`vision_logger.py`](vision_logger.py):

| Parameter | Default | Purpose |
|---|---|---|
| `MODEL_NAME` | `gemma4:12b` | Ollama model tag |
| `INTERVAL_SECONDS` | `10` | Minimum seconds between VLM analyses |
| `MOTION_THRESHOLD` | `500` | Changed-pixel count to trigger motion |
| `DIFF_THRESHOLD` | `25` | Per-pixel intensity delta to count as "changed" |

## Dependencies

Two packages in [`requirements.txt`](requirements.txt):

- `ollama` — Python client for the Ollama local inference server
- `opencv-python` — Webcam capture and frame differencing

Requires [Ollama](https://ollama.com) installed and running as a system service.

> **Note:** The Gemma 4 12B model weights (~8 GB) are downloaded on first use via `ollama pull gemma4:12b`.

## System Requirements

- macOS with Apple Silicon (M1/M2/M3/M4/M5)
- 16 GB+ unified memory recommended for the 12B model
- Python 3.10+