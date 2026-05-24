# Start MehvishLog Project with MLX-VLM

This plan outlines the steps to build a local AI vision logger on your MacBook Air M4 under the `MehvishLog` project directory.

## Proposed Changes

### Setup Environment & Directories
#### [NEW] `/Users/afsalaazeez/Workspace/github/afsalaazeez/MehvishLog/`
Create the project directory.

#### [NEW] `/Users/afsalaazeez/Workspace/github/afsalaazeez/MehvishLog/requirements.txt`
Dependencies file including Apple's MLX-VLM and OpenCV.
```txt
mlx-vlm
opencv-python
```

### Application Code
#### [NEW] `/Users/afsalaazeez/Workspace/github/afsalaazeez/MehvishLog/vision_logger.py`
The AI vision logger script that will:
1. Initialize the `mlx-community/Qwen2-VL-2B-Instruct-4bit` model using `mlx-vlm`.
2. Capture the webcam feed using OpenCV.
3. Keep track of previous frames to detect sudden/significant motion using basic frame differencing.
4. Analyze the frame every 5 seconds **only if motion is detected**.
5. Save a one-sentence descriptive log with a timestamp to `m4_vision_log.txt`.

#### [NEW] `/Users/afsalaazeez/Workspace/github/afsalaazeez/MehvishLog/m4_vision_log.txt`
The automatically generated text file where the logs will be written.

## Open Questions
- For "significant movement", I will implement a basic frame differencing algorithm with a configurable threshold. Let me know if you would like me to use a specific approach instead (like background subtraction).
- The script will run continuously in the terminal until stopped with `Ctrl+C`. Let me know if you prefer to run it in a specific loop count instead for testing.

## Verification Plan
### Execution Steps
- I will run `python3 -m venv venv` and configure the environment.
- Install dependencies using `pip install mlx-vlm opencv-python`.
- Start the `vision_logger.py` script in the terminal and verify that:
  - It successfully requests and accesses your webcam.
  - The model downloads and caches correctly from HuggingFace.
  - `m4_vision_log.txt` gets updated.
