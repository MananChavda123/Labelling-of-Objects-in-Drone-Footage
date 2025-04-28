import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
from pathlib import Path

# Define augmentation pipeline
augmentation = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.RandomCrop(width=640, height=640, p=0.5),
    A.ColorJitter(p=0.2),
    A.Rotate(limit=30, p=0.5),
    A.Blur(blur_limit=3, p=0.3),
    ToTensorV2()
])

def augment_images(input_img_dir, input_label_dir, output_img_dir, output_label_dir, exclude_classes=[1, 4]):
    """
    Augment the images and labels in the dataset. 
    Only augment for classes not in `exclude_classes` (pedestrian and car).
    """
    image_paths = list(Path(input_img_dir).rglob("*.jpg"))
    label_paths = list(Path(input_label_dir).rglob("*.txt"))

    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)
    
    if not os.path.exists(output_label_dir):
        os.makedirs(output_label_dir)

    # Iterate over image-label pairs
    for img_path, label_path in zip(image_paths, label_paths):
        # Load the image and labels
        img = cv2.imread(str(img_path))
        with open(label_path, 'r') as f:
            labels = f.readlines()
        
        # Filter labels for the classes to exclude (pedestrian and car)
        filtered_labels = [label for label in labels if int(label.split()[0]) not in exclude_classes]
        
        if filtered_labels:
            # Apply augmentation
            augmented = augmentation(image=img)
            augmented_img = augmented['image']
            
            # Save the augmented image
            new_img_path = os.path.join(output_img_dir, img_path.name)
            cv2.imwrite(new_img_path, augmented_img)

            # Create new label path and save
            new_label_path = os.path.join(output_label_dir, label_path.name)
            with open(new_label_path, 'w') as new_label_file:
                new_label_file.writelines(filtered_labels)

# Call the function to augment data for all classes except 'pedestrian' (1) and 'car' (4)
augment_images(
    input_image_dir = Path('C:/Users/havis/OneDrive/Desktop/vision-drone-project/datasets/VisDrone/VisDrone2019-DET_train/images'),
    input_label_dir = Path('C:/Users/havis/OneDrive/Desktop/vision-drone-project/datasets/VisDrone/VisDrone2019-DET_train/labels'),
    output_image_dir = Path('C:/Users/havis/OneDrive/Desktop/vision-drone-project/dataset/images/train'),
    output_label_dir = Path('C:/Users/havis/OneDrive/Desktop/vision-drone-project/dataset/labels/train'),
    exclude_classes=[1, 4]  # Classes to exclude from augmentation: pedestrian (1) and car (4)
)
