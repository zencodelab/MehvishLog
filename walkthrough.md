# MehvishLog: Local AI Vision Logger Walkthrough

Welcome to your new local AI Vision Logger! I've successfully built the project adhering entirely to your constraints: utilizing the power of your MacBook Air M4 by integrating Apple's `mlx-vlm` and using OpenCV for battery-saving motion detection.

## What was built
1. **Virtual Environment Setup**: Maintained inside `/MehvishLog/venv` to avoid polluting your global environment.
2. **Apple MLX Integration**: Using `mlx-vlm` and the `mlx-community/Qwen2-VL-2B-Instruct-4bit` model to run fast visual analysis entirely off your M4's unified memory.
3. **Motion Sensitivity via OpenCV**: The script compares successive frames using an absolute diff map and basic thresholding. This ensures the model is *only* called when you are actively moving in front of the camera, saving valuable battery and compute cycles. 
4. **vision_logger.py**: This central orchestrator handles the feed, triggers the generation, and logs everything to `m4_vision_log.txt`.

> [!TIP]
> The threshold variables `INTERVAL_SECONDS`, `MOTION_THRESHOLD`, and `DIFF_THRESHOLD` inside `vision_logger.py` are all cleanly defined at the top of the script so you can tweak the motion sensitivity easily!

## Verification Results
I've started the model download for `Qwen2-VL-2B` using the script to verify the workflow is sound. However, **you must execute the script in your own terminal window** at this point! 

macOS requires you to explicitly grant the Terminal access to the Camera. If I run it in the background, you'll never see the prompt.

## How to Run It Live

1. Open your integrated VS Code / Mac terminal.
2. Navigate to the project root:
   ```bash
   cd /Users/afsalaazeez/Workspace/github/afsalaazeez/MehvishLog
   ```
3. Activate the environment:
   ```bash
   source venv/bin/activate
   ```
4. Run the logger!
   ```bash
   python vision_logger.py
   ```

> [!IMPORTANT]
> The very first time you run this script, macOS will pop up a window asking for Camera Permissions. Be sure to click "Allow". 
> Also note that the first launch may take an extra 1-3 minutes because it needs to finish downloading the 1.5GB compiled LLM weights from HuggingFace to your local cache. It will be instant on subsequent runs!

## Code Delivered
[vision_logger.py](file:///Users/afsalaazeez/Workspace/github/afsalaazeez/MehvishLog/vision_logger.py)
