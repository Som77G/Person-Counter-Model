import cv2
import numpy as np
import onnxruntime as ort


class PersonDetector:
    def __init__(self, model_path, conf_threshold):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.conf_threshold = conf_threshold

        self.input_name = self.session.get_inputs()[0].name
        self.input_size = 640  # YOLOv8 default

    def preprocess(self, frame):
        img = cv2.resize(frame, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
        img = np.expand_dims(img, axis=0).astype(np.float32)
        return img

    def postprocess(self, outputs, original_shape):
        preds = outputs[0]
        h, w = original_shape

        # Debug: print shape before any transformation
        # print(f"[DEBUG] Raw output shape: {outputs[0].shape}")
        # print(f"[DEBUG] Conf threshold: {self.conf_threshold}")

        # Handle both possible shapes
        if preds.ndim == 3 and preds.shape[1] < preds.shape[2]:
            # Shape is (1, 84, 8400) → transpose to (8400, 84)
            preds = preds[0].T
        else:
            preds = preds[0]  # Already (8400, 85)

        # Debug: max score using correctly shaped preds
        # print(f"[DEBUG] Max score across all preds: {preds[:, 4:].max():.4f}")

        boxes = []
        scores = []  # ← track scores here, parallel to boxes

        for pred in preds:
            class_scores = pred[4:]
            cls = np.argmax(class_scores)
            score = class_scores[cls]

            if cls == 0 and score > self.conf_threshold:
                x, y, bw, bh = pred[:4]
                x1 = int((x - bw / 2) * w / self.input_size)
                y1 = int((y - bh / 2) * h / self.input_size)
                x2 = int((x + bw / 2) * w / self.input_size)
                y2 = int((y + bh / 2) * h / self.input_size)
                boxes.append((x1, y1, x2, y2))
                scores.append(float(score))  # ← append score alongside box

        # print(f"[DEBUG] Boxes before NMS: {len(boxes)}")

        # Apply NMS only if we have detections
        if boxes:
            rects = np.array([[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in boxes], dtype=np.float32)
            indices = cv2.dnn.NMSBoxes(rects.tolist(), scores, self.conf_threshold, 0.4)
            boxes = [boxes[i] for i in indices.flatten()]

        # print(f"[DEBUG] Boxes after NMS: {len(boxes)}")
        return boxes

    def detect(self, frame):
        original_shape = frame.shape[:2]

        input_tensor = self.preprocess(frame)
        outputs = self.session.run(None, {self.input_name: input_tensor})

        detections = self.postprocess(outputs, original_shape)

        return detections