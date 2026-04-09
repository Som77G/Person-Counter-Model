import cv2
import time
from stream.capture import StreamCapture
from detection.detector import PersonDetector
from utils.counter import PersonCounter
import config


def resize_frame(frame, width=640):
    h, w = frame.shape[:2]
    aspect_ratio = h / w
    new_height = int(width * aspect_ratio)
    return cv2.resize(frame, (width, new_height))


def main():
    print("Starting Person Counter Service...")

    stream = StreamCapture(config.RTSP_URL)
    detector = PersonDetector(config.MODEL_PATH, config.CONFIDENCE_THRESHOLD)
    counter = PersonCounter()

    frame_count = 0

    try:
        while True:
            frame = stream.get_frame()

            if frame is None:
                print("Stream error. Retrying in 5 seconds...")
                time.sleep(5)
                continue

            frame = resize_frame(frame)

            detections = detector.detect(frame)
            count = counter.update(detections)

            frame_count += 1

            if frame_count % 10 == 0:
                print(f"[INFO] Persons detected: {count}")

    except KeyboardInterrupt:
        print("\n Stopping service (Ctrl+C pressed)...")

    finally:
        print("🔌 Releasing stream...")
        stream.release()

if __name__ == "__main__":
    main()