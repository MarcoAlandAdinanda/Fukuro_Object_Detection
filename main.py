import cv2
import time
import numpy as np
import torch
import torchvision
from box_fast_rcnn import BoxFRCNN
from realsense_camera import *
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
WEIGHTS_FILE = "./V2_Mobile_faster_rcnn_state.pth"

num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Load the traines weights
model.load_state_dict(torch.load(WEIGHTS_FILE))
model = model.to(device)
model.eval()

# For detection on video, replace the 0 with video path 
cap = cv2.VideoCapture(0) 

# Check if the video capture is opened
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Detector class and realsense
detector = BoxFRCNN(model, device)
rs = RealsenseCamera()

fps_list = []
prev_time = 0

while True:
    # Get frame in real time from Realsense camera
    ret, bgr_frame, depth_frame = rs.get_frame_stream()

    # Detect the bgr frame
    labels, boxes, scores = detector.obj_detector(bgr_frame)

    for label, box, score in zip(labels, boxes, scores):
        x, y, w, h = box
        cx = x + w // 2
        cy = y + h // 2
        depth_mm = depth_frame[cy, cx]

        cv2.rectangle(bgr_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(bgr_frame, f"{label}: {score:.2f}; Distance: {depth_mm / 10} cm", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        print(f"Center point of the box: {depth_mm / 10} cm")

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    fps_list.append(fps)

    # Display FPS
    cv2.putText(bgr_frame, f"FPS: {int(np.mean(fps_list[-30:]))}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Object Detection', bgr_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

rs.release()
cv2.destroyAllWindows()