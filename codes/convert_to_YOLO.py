import os
import glob
import cv2
import tqdm

# Paths (Change these)
visdrone_path = "C:/Users/Manan/Desktop/Object_Deection_DL/drone_project/datasets/VisDrone2019-DET-val/VisDrone2019-DET-val"
image_folder = os.path.join(visdrone_path, "images")
annotation_folder = os.path.join(visdrone_path, "annotations")
output_label_folder = os.path.join(visdrone_path, "labels")

# Create output folder if not exists
os.makedirs(output_label_folder, exist_ok=True)

# Convert each annotation file
for txt_file in tqdm.tqdm(glob.glob(os.path.join(annotation_folder, "*.txt"))):
    filename = os.path.basename(txt_file)
    image_path = os.path.join(image_folder, filename.replace(".txt", ".jpg"))

    # Load image to get width and height
    image = cv2.imread(image_path)
    if image is None:
        print(f"Skipping {filename}: Image not found")
        continue
    h, w, _ = image.shape

    # Process annotation file
    with open(txt_file, "r") as f, open(os.path.join(output_label_folder, filename), "w") as yolo_f:
        for line in f:
            line = line.strip()
            if not line:  # Skip empty lines
                continue  

            values = line.split(",")  
            if len(values) < 6:  # Ensure enough values
                print(f"Skipping malformed line in {filename}: {line}")
                continue  

            try:
                values = list(map(int, values[:6]))  # Convert only first 6 values to int
                class_id = values[5]  # Fix: Use correct index for class_id
            except ValueError:
                print(f"Skipping non-integer values in {filename}: {line}")
                continue  

            if class_id == 0:
                continue  # Ignore class 0

            # Fix: Convert class_id from 1-10 to 0-9
            class_id -= 1  
            if class_id > 9:  
                print(f"Skipping invalid class ID {class_id+1} in {filename}")  
                continue  

            # Fix: Use correct bounding box indices
            xmin, ymin, box_w, box_h = values[0:4]
            x_center = (xmin + box_w / 2) / w
            y_center = (ymin + box_h / 2) / h
            box_w = box_w / w
            box_h = box_h / h

            # Fix: Format numbers to 6 decimal places
            yolo_f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")

print("✅ Conversion complete! YOLO labels saved in:", output_label_folder)
