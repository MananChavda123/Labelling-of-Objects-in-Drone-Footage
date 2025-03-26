import torch
print(torch.cuda.is_available())  # Should print True if CUDA is detected
print(torch.cuda.get_device_name(0))  # Shows GPU name if available


yolo task=detect mode=val model=C:/Users/Manan/Desktop/Object_Deection_DL/drone_project/codes/runs/detect/train4/weights/best.pt data=C:/Users/Manan/Desktop/Object_Deection_DL/drone_project/datasets/VisDrone.yaml

yolo task=detect mode=predict model=C:/Users/Manan/Desktop/Object_Deection_DL/drone_project/codes/runs/detect/train6/weights/best.pt source=C:/Users/Manan/Desktop/Object_Deection_DL/drone_project/datasets/test-image-1.jpeg
