import cv2
import time
import datetime
import base64
import ollama

# Configuration
MODEL_NAME = "gemma4:12b"
LOG_FILE = "m4_vision_log.txt"
INTERVAL_SECONDS = 10    # Seconds between VLM analyses (10s recommended for 12B model)
MOTION_THRESHOLD = 500   # Number of changed pixels to trigger motion
DIFF_THRESHOLD = 25      # Intensity difference to consider a pixel changed

PROMPT = "Describe this scene accurately in one brief sentence."


def encode_frame(frame):
    """Encode an OpenCV frame to base64 JPEG for the Ollama API."""
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode("utf-8")


def describe_frame(frame):
    """Send a frame to Gemma 4 via Ollama and return the description."""
    image_b64 = encode_frame(frame)
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": PROMPT,
                "images": [image_b64],
            }
        ],
    )
    return response.message.content.replace("\n", " ").strip()


def main():
    # Verify Ollama connection and model availability
    print(f"Connecting to Ollama (model: {MODEL_NAME})...")
    try:
        ollama.show(MODEL_NAME)
        print("Model ready.")
    except ollama.ResponseError:
        print(f"Model '{MODEL_NAME}' not found. Pulling it now (this may take a few minutes)...")
        ollama.pull(MODEL_NAME)
        print("Model pulled successfully.")

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

                try:
                    description = describe_frame(frame)

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
