class PersonCounter:
    def __init__(self):
        self.count = 0

    def update(self, detections):
        self.count = len(detections)
        return self.count