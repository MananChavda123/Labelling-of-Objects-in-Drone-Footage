import os
import shutil
import random

# Source directories (VisDrone full training set)
images_dir = 'C:/Users/havis/OneDrive/Desktop/vision-drone-project/datasets/VisDrone/VisDrone2019-DET-train/images'
annotations_dir = 'C:/Users/havis/OneDrive/Desktop/vision-drone-project/datasets/VisDrone/VisDrone2019-DET-train/annotations'

# Target directories for YOLOv5 format
base_output = 'C:/Users/havis/OneDrive/Desktop/vision-drone-project/dataset'
train_images_out = os.path.join(base_output, 'images/train')
train_labels_out = os.path.join(base_output, 'labels/train')
val_images_out = os.path.join(base_output, 'images/val')
val_labels_out = os.path.join(base_output, 'labels/val')

# Create output dirs
for path in [train_images_out, train_labels_out, val_images_out, val_labels_out]:
    os.makedirs(path, exist_ok=True)

# All images in the train set
image_files = [f for f in os.listdir(images_dir) if f.lower().endswith('.jpg')]
random.shuffle(image_files)

# Define split ratio
val_ratio = 0.2
val_count = int(len(image_files) * val_ratio)

# Split into val and train
val_files = image_files[:val_count]
train_files = image_files[val_count:]

# Copying files
def copy_files(file_list, target_img_dir, target_lbl_dir):
    for img_name in file_list:
        label_name = img_name.replace('.jpg', '.txt')

        # Copy image
        src_img = os.path.join(images_dir, img_name)
        dst_img = os.path.join(target_img_dir, img_name)
        shutil.copy2(src_img, dst_img)

        # Copy label
        src_lbl = os.path.join(annotations_dir, label_name)
        dst_lbl = os.path.join(target_lbl_dir, label_name)
        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, dst_lbl)

# Do the copying
copy_files(train_files, train_images_out, train_labels_out)
copy_files(val_files, val_images_out, val_labels_out)

print(f"✅ Split complete. {len(train_files)} training images and {len(val_files)} validation images copied.")
