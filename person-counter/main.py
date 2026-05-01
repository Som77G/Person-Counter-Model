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

            # ---------- FRAME START ----------
            frame_start = time.time()

            # ---------- PREPROCESS ----------
            t0 = time.time()
            frame = resize_frame(frame)
            t1 = time.time()

            preprocess_latency = (t1 - t0) * 1000

            # ---------- INFERENCE ----------
            t2 = time.time()
            detections = detector.detect(frame)
            t3 = time.time()

            inference_latency = (t3 - t2) * 1000

            # ---------- POSTPROCESS ----------
            t4 = time.time()
            count = counter.update(detections)
            frame = draw_boxes(frame, detections, count)
            t5 = time.time()

            postprocess_latency = (t5 - t4) * 1000

                        # ---------- ENCODE / QUEUE ----------
            t6 = time.time()
            if not frame_queue.full():
                frame_queue.put(frame)
            t7 = time.time()

            encode_latency = (t7 - t6) * 1000

            # ---------- END-TO-END ----------
            frame_end = time.time()
            total_latency = (frame_end - frame_start) * 1000

            # ---------- FPS ----------
            fps = 1.0 / (frame_end - frame_start)

            # ---------- STORE ----------
            latency_history.append(total_latency)

            cv2.putText(frame, f"RTT*: {total_latency:.1f} ms",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.putText(frame, f"Inf: {inference_latency:.1f} ms",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.putText(frame, f"Pre: {preprocess_latency:.1f} ms",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.putText(frame, f"Post: {postprocess_latency:.1f} ms",
                        (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            cv2.putText(frame, f"Enc: {encode_latency:.1f} ms",
                        (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 2)

            cv2.putText(frame, f"FPS: {fps:.2f}",
                        (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 150), 2)
            
            if frame_id % 30 == 0:
                print(
                    f"[INFO] Count: {count} | "
                    f"RTT*: {total_latency:.1f} ms | "
                    f"Inf: {inference_latency:.1f} ms | "
                    f"Pre: {preprocess_latency:.1f} ms | "
                    f"Post: {postprocess_latency:.1f} ms | "
                    f"Enc: {encode_latency:.1f} ms | "
                    f"FPS: {fps:.2f}"
                )
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