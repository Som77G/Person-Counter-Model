import os

RTSP_URL = os.getenv("RTSP_URL", "rtsp://admin:admin123@10.45.0.4:554/avstream/channel=1/stream=0.sdp")
# RTSP_URL = 0

MODEL_PATH = "yolov8n.onnx"
CONFIDENCE_THRESHOLD = 0.4
INPUT_SIZE = 640


