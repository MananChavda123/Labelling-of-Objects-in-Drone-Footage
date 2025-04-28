import cv2
import os
# === Set paths ===
image_dir = 'C:/Users/havis/OneDrive/Desktop/vision-drone-project/datasets/VisDrone/VisDrone2019-DET-train/images'
annotation_dir = 'C:/Users/havis/OneDrive/Desktop/vision-drone-project/datasets/VisDrone/VisDrone2019-DET-train/annotations'

# Class names for YOLO (1-10 in VisDrone → 0-9 in YOLO)
class_names = ['pedestrian', 'people', 'bicycle', 'car', 'van',
               'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']

# Load first 5 image files
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])[:5]

for img_file in image_files:
    img_path = os.path.join(image_dir, img_file)
    ann_path = os.path.join(annotation_dir, img_file.replace('.jpg', '.txt'))

    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not load {img_file}")
        continue

    if not os.path.exists(ann_path):
        print(f"No annotation for {img_file}")
        continue

    with open(ann_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 7:
                continue

            left, top, width, height = map(int, parts[0:4])
            cls_id = int(parts[5])

            # Skip categories 0 and 11 (ignored / others)
            if cls_id in [0, 11]:
                continue

            yolo_cls_id = cls_id - 1
            label = class_names[yolo_cls_id]

            # Draw bbox
            cv2.rectangle(img, (left, top), (left + width, top + height), (0, 255, 0), 2)
            cv2.putText(img, label, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Show image
    cv2.imshow(f"Annotated - {img_file}", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
