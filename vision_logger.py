import cv2
import time
import datetime
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

# Configuration
MODEL_PATH = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
LOG_FILE = "m4_vision_log.txt"
IMAGE_PATH = "/tmp/vision_logger_frame.jpg"
INTERVAL_SECONDS = 5
MOTION_THRESHOLD = 500  # Number of changed pixels to trigger motion
DIFF_THRESHOLD = 25     # Intensity difference to consider a pixel changed

def main():
    print(f"Loading MLX-VLM model: {MODEL_PATH}...")
    model, processor = load(MODEL_PATH)
    print("Model loaded successfully.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Webcam initialized. Monitoring for motion (Ctrl+C to stop)...")

    last_process_time = 0
    prev_gray = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                time.sleep(1)
                continue
            
            # Convert to grayscale and blur to reduce noise
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            motion_detected = False
            
            if prev_gray is not None:
                # Compute absolute difference
                frame_delta = cv2.absdiff(prev_gray, gray)
                thresh = cv2.threshold(frame_delta, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
                thresh = cv2.dilate(thresh, None, iterations=2)
                
                # Check how much changed
                non_zero_count = cv2.countNonZero(thresh)
                if non_zero_count > MOTION_THRESHOLD:
                    motion_detected = True
                    
            prev_gray = gray
            
            current_time = time.time()
            
            # Only process if motion detected AND interval elapsed
            if motion_detected and (current_time - last_process_time) >= INTERVAL_SECONDS:
                print("Motion detected. Processing frame...")
                
                # Save frame to disk for the model
                cv2.imwrite(IMAGE_PATH, frame)
                
                # Generate description
                raw_prompt = "Describe this scene accurately in one brief sentence."
                prompt = apply_chat_template(
                    processor,
                    getattr(model, "config", {}),
                    raw_prompt,
                    num_images=1
                )
                try:
                    output = generate(
                        model=model,
                        processor=processor,
                        prompt=prompt,
                        image=IMAGE_PATH,
                        max_tokens=60,
                        verbose=False
                    )
                    
                    # Clean output
                    description = output.replace('\n', ' ').strip()
                    
                    # Log to file
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_entry = f"[{timestamp}] {description}\n"
                    
                    with open(LOG_FILE, "a") as f:
                        f.write(log_entry)
                        
                    print(f"Logged: {log_entry.strip()}")
                    
                except Exception as e:
                    print(f"Error during VLM generation: {e}")
                    
                last_process_time = current_time
                
    except KeyboardInterrupt:
        print("\nStopping vision logger...")
    finally:
        cap.release()
        print("Webcam released.")

if __name__ == "__main__":
    main()
