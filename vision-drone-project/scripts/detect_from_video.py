import os
import cv2
import torch
import numpy as np
from pathlib import Path
import sys
import torchvision
sys.path.append('C:/Users/havis/OneDrive/Desktop/vision-drone-project/yolov5')

# Custom utility functions to replace YOLOv5 utils
def select_device(device='cpu'):
    return torch.device(device if torch.cuda.is_available() else 'cpu')

def non_max_suppression(pred, conf_thres=0.15, iou_thres=0.45):
    xc = pred[..., 4] > conf_thres  # candidates with confidence > threshold
    output = [None] * pred.shape[0]
    for i, x in enumerate(pred):
        x = x[xc[i]]  # Filter candidates
        if len(x):
            boxes = x[:, :4]
            scores = x[:, 4]
            classes = x[:, 5]
            keep = torchvision.ops.nms(boxes, scores, iou_thres)
            output[i] = x[keep]
    return output

def scale_coords(img_shape, coords, img_size, ratio_pad=None):
    gain = min(img_size[0] / img_shape[0], img_size[1] / img_shape[1])
    pad = [(img_size[1] - img_shape[1] * gain) / 2, (img_size[0] - img_shape[0] * gain) / 2]
    pad = torch.tensor(pad, device=coords.device)  # Convert pad to tensor and move to the correct device

    # Expand pad to match the number of coordinates (4 values: x1, y1, x2, y2)
    pad = pad.repeat(2)  # Repeat the padding values for both x and y directions

    coords = coords - pad
    coords /= gain
    return coords



def letterbox(img, new_shape=640, color=(114, 114, 114), stride=32):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img

# Paths
weights = 'yolov5/runs/train/exp11/weights/best.pt'
video_path = '3121459-uhd_3840_2160_24fps.mp4'
output_img_dir = 'autolabel/images'
output_lbl_dir = 'autolabel/labels'

# Check if video is loaded
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Couldn't open video file.")
    exit()

frame_id = 0
os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_lbl_dir, exist_ok=True)

# Load model
device = select_device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights, force_reload=True)
model.to(device).eval()

def convert_to_yolo_format(xyxy, w, h):
    # Extract only the first 4 values (x1, y1, x2, y2)
    box = xyxy[:4]

    if len(box) != 4:
        raise ValueError(f"Expected 4 values in xyxy, but got {len(box)}: {box}")

    # Unpack the bounding box coordinates
    x1, y1, x2, y2 = box

    # Convert to YOLO format
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    width = x2 - x1
    height = y2 - y1

    # Normalize coordinates to be between 0 and 1
    x_center /= w
    y_center /= h
    width /= w
    height /= h

    return x_center, y_center, width, height



while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame, ending video processing.")
        break

    print(f"Processing frame {frame_id}")

    # Preprocess
    img = letterbox(frame, new_shape=640)  # Only capture the resized image

    # Check the image shape
    print(f"Image shape after letterbox: {img.shape}")  # Should print something like (640, 640, 3)

    # Convert BGR to RGB (if needed)
    img = img[..., ::-1]  # Convert BGR to RGB

    # Ensure correct shape (H, W, C) -> (C, H, W)
    if img.ndim == 3:  # If the image has 3 dimensions (height, width, channels)
        img = np.transpose(img, (2, 0, 1))  # Convert from (H, W, C) to (C, H, W)
    else:
        print("Error: Image does not have the expected 3D shape.")

    # Make sure the array is contiguous in memory
    img = np.ascontiguousarray(img)

    # Convert image to tensor
    img_tensor = torch.from_numpy(img).to(device).float()  # Convert NumPy array to tensor
    img_tensor /= 255.0  # Normalize to [0, 1]

    # Add batch dimension if necessary
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension: (C, H, W) -> (1, C, H, W)

    # Now the img_tensor is ready for inference


    # Inference
    with torch.no_grad():
        pred = model(img_tensor)
        print(f"Prediction Tensor: {pred}")
        pred = non_max_suppression(pred, conf_thres=0.15, iou_thres=0.45)[0]

    h, w = frame.shape[:2]
    label_lines = []

    if pred is not None and len(pred):
        print(f"Detections found: {len(pred)}")
        pred[:, :4] = scale_coords(img_tensor.shape[2:], pred[:, :4], frame.shape).round()
        for *xyxy, conf, cls in pred.tolist():
            x_center, y_center, width, height = convert_to_yolo_format(xyxy, w, h)
            label_lines.append(f"{int(cls)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    else:
        print("No detections in this frame")

    # Save frame and label
    img_path = os.path.join(output_img_dir, f"frame_{frame_id:05}.jpg")
    label_path = os.path.join(output_lbl_dir, f"frame_{frame_id:05}.txt")

    print(f"Saving frame to: {img_path}")
    print(f"Saving label to: {label_path}")

    if cv2.imwrite(img_path, frame):
        print(f"Saved frame {frame_id}")
    else:
        print(f"Failed to save frame {frame_id}")

    if len(label_lines) > 0:
        with open(label_path, 'w') as f:
            f.write('\n'.join(label_lines))
    else:
        print(f"No labels for frame {frame_id}")

    frame_id += 1

cap.release()
print("✅ Video processed and auto-labeled.")
