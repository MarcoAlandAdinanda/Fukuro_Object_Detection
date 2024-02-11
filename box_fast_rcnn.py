import cv2
import torch
import numpy as np

class BoxFRCNN:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.detection_threshold = 0.7
        
    def obj_detector(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
        frame /= 255.0
        frame = torch.from_numpy(frame)
        frame = frame.unsqueeze(0)
        frame = frame.permute(0, 3, 1, 2).to(self.device)

        with torch.no_grad():
            output = self.model(frame)

        boxes = output[0]['boxes'].data.cpu().numpy()
        scores = output[0]['scores'].data.cpu().numpy()
        labels = output[0]['labels'].data.cpu().numpy()

        labels = labels[scores >= self.detection_threshold]
        boxes = boxes[scores >= self.detection_threshold].astype(np.int32)
        scores = scores[scores >= self.detection_threshold]

        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        
        return labels, boxes, scores
        