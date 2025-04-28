import os

# Append new auto-labeled data
# You can move images/labels from `autolabel/` to `dataset/images/train` and `labels/train`

os.system("python yolov5/train.py --img 640 --batch 16 --epochs 50 --data data.yaml --weights yolov5/runs/train/visdrone-train/weights/best.pt --name visdrone-retrain")
