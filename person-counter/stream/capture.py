import cv2

class StreamCapture:
    def __init__(self, source):
        # If source is int → webcam
        if isinstance(source, int):
            self.cap = cv2.VideoCapture(source)
        else:
            self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        self.cap.release()