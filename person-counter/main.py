import cv2
import time
import subprocess
import threading
import queue
import matplotlib.pyplot as plt
from collections import deque

from stream.capture import StreamCapture
from detection.detector import PersonDetector
from utils.counter import PersonCounter
import config


# ================= CONFIG =================
FRAME_SKIP = 3
OUTPUT_WIDTH = 480
OUTPUT_FPS = 10


# ================= LATENCY =================
latency_history = deque(maxlen=100)


# ================= UTIL ====================
def resize_frame(frame, width=OUTPUT_WIDTH):
    h, w = frame.shape[:2]
    aspect_ratio = h / w
    new_height = int(width * aspect_ratio)
    return cv2.resize(frame, (width, new_height))


def draw_boxes(frame, detections, count):
    for (x1, y1, x2, y2) in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(frame, f"Count: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame


# ================= FFMPEG ==================
def start_ffmpeg(width, height):
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(OUTPUT_FPS),
        "-i", "-",
        "-an",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-crf", "30",
        "-f", "rtsp",
        "-rtsp_transport", "tcp",
        config.OUTPUT_RTSP_URL
    ]

    return subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)


# ================= GRAPH ===================
def plot_latency():
    plt.ion()
    fig, ax = plt.subplots()

    while True:
        if len(latency_history) > 0:
            ax.clear()
            ax.plot(list(latency_history))
            ax.set_title("End-to-End Latency (ms)")
            ax.set_xlabel("Frames")
            ax.set_ylabel("Latency")
            plt.pause(0.1)


# ================= MAIN ====================
def main():
    print("Starting Person Counter Service with Latency Tracking...")

    stream = StreamCapture(config.RTSP_URL)
    detector = PersonDetector(config.MODEL_PATH, config.CONFIDENCE_THRESHOLD)
    counter = PersonCounter()

    frame_queue = queue.Queue(maxsize=5)
    ffmpeg_process = None

    # ---------- Start graph thread ----------
    threading.Thread(target=plot_latency, daemon=True).start()

    # ---------- FFmpeg writer thread ----------
    def ffmpeg_writer():
        nonlocal ffmpeg_process

        while True:
            frame = frame_queue.get()

            if frame is None:
                break

            if ffmpeg_process is None:
                h, w = frame.shape[:2]
                ffmpeg_process = start_ffmpeg(w, h)
                print("[INFO] FFmpeg started")

            try:
                ffmpeg_process.stdin.write(frame.tobytes())
            except (BrokenPipeError, OSError):
                print("[WARN] FFmpeg crashed. Restarting...")
                ffmpeg_process = None

    threading.Thread(target=ffmpeg_writer, daemon=True).start()

    frame_id = 0

    try:
        while True:
            frame = stream.get_frame()

            if frame is None:
                print("[WARN] Stream error. Retrying...")
                time.sleep(2)
                continue

            frame_id += 1

            # ---------- FRAME SKIP ----------
            if frame_id % FRAME_SKIP != 0:
                continue

            # ---------- CAPTURE TIME ----------
            capture_time = time.time()

            frame = resize_frame(frame)

            # ---------- PROCESSING LATENCY ----------
            t1 = time.time()
            detections = detector.detect(frame)
            t2 = time.time()

            processing_latency = (t2 - t1) * 1000  # ms

            count = counter.update(detections)

            # ---------- DRAW ----------
            frame = draw_boxes(frame, detections, count)

            # ---------- TOTAL LATENCY ----------
            current_time = time.time()
            total_latency = (current_time - capture_time) * 1000
            network_latency = total_latency - processing_latency

            latency_history.append(total_latency)

            # ---------- OVERLAY ----------
            cv2.putText(frame, f"Total: {total_latency:.1f} ms",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.putText(frame, f"Proc: {processing_latency:.1f} ms",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.putText(frame, f"Net: {network_latency:.1f} ms",
                        (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # ---------- QUEUE ----------
            if not frame_queue.full():
                frame_queue.put(frame)

            if frame_id % 30 == 0:
                print(f"[INFO] Count: {count} Total: {total_latency:.1f} ms | Proc: {processing_latency:.1f} ms")

    except KeyboardInterrupt:
        print("\n Stopping service...")

    finally:
        stream.release()
        frame_queue.put(None)

        if ffmpeg_process:
            ffmpeg_process.stdin.close()
            ffmpeg_process.wait()

        print("Clean shutdown complete")


if __name__ == "__main__":
    main()