import os

RTSP_URL = os.getenv("RTSP_URL", "rtsp://10.45.0.2:8554/live")
# RTSP_URL = "rtsp://10.45.0.2:8554/live"

MODEL_PATH = "yolov8n.onnx"
CONFIDENCE_THRESHOLD = 0.4
INPUT_SIZE = 640

OUTPUT_RTSP_URL = "rtsp://10.45.0.1:8554/output"


