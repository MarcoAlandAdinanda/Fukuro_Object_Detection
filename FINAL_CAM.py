import cv2
import time
import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
WEIGHTS_FILE = "./faster_rcnn_state.pth"

num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Load the traines weights
model.load_state_dict(torch.load(WEIGHTS_FILE))
model = model.to(device)
model.eval()

# video_path = "./.../..."

# For detection on video, replace the 0 with video path 
cap = cv2.VideoCapture(0) 

# Check if the video capture is opened
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Object detection function
def obj_detector(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img /= 255.0
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    img = img.permute(0, 3, 1, 2).to(device)

    detection_threshold = 0.70

    with torch.no_grad():
        output = model(img)

    boxes = output[0]['boxes'].data.cpu().numpy()
    scores = output[0]['scores'].data.cpu().numpy()
    labels = output[0]['labels'].data.cpu().numpy()

    labels = labels[scores >= detection_threshold]
    boxes = boxes[scores >= detection_threshold].astype(np.int32)
    scores = scores[scores >= detection_threshold]

    boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
    boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

    return labels, boxes, scores

fps_list = []
prev_time = 0

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("End of video")
        break

    labels, boxes, scores = obj_detector(frame)

    for label, box, score in zip(labels, boxes, scores):
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{label}: {score:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        center_x = x + w // 2
        center_y = y + h // 2

        print(f"Center point of the box: ({center_x}, {center_y})")

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    fps_list.append(fps)

    # Display FPS
    cv2.putText(frame, f"FPS: {int(np.mean(fps_list[-30:]))}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
