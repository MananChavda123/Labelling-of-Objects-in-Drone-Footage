import os
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def convert_box(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2] / 2.0) * dw
    y = (box[1] + box[3] / 2.0) * dh
    w = box[2] * dw
    h = box[3] * dh
    return x, y, w, h

def convert_visdrone_to_yolo(data_dir, output_dir):
    image_dir = Path(data_dir) / 'images'
    anno_dir = Path(data_dir) / 'annotations'
    
    # Set the label directory paths
    label_dir_train = Path(output_dir) / 'labels' / 'train'  # Save labels outside 'datasets'
    label_dir_val = Path(output_dir) / 'labels' / 'val'      # Save labels outside 'datasets'
    label_dir_train.mkdir(parents=True, exist_ok=True)
    label_dir_val.mkdir(parents=True, exist_ok=True)
    
    # Set the image directory paths
    image_dir_train = Path(output_dir) / 'images' / 'train'
    image_dir_val = Path(output_dir) / 'images' / 'val'
    image_dir_train.mkdir(parents=True, exist_ok=True)
    image_dir_val.mkdir(parents=True, exist_ok=True)

    for anno_file in tqdm(list(anno_dir.glob('*.txt')), desc=f"Converting {data_dir}"):
        image_file = image_dir / anno_file.name.replace('.txt', '.jpg')
        if not image_file.exists():
            continue

        # Determine if the image belongs to the train or val set
        if 'train' in str(anno_file):
            image_dest = image_dir_train / anno_file.name.replace('.txt', '.jpg')
            label_dest = label_dir_train / anno_file.name
        else:
            image_dest = image_dir_val / anno_file.name.replace('.txt', '.jpg')
            label_dest = label_dir_val / anno_file.name

        # Copy the image to the destination folder
        shutil.copy(image_file, image_dest)

        img = Image.open(image_file)
        img_size = img.size  # (width, height)

        yolo_labels = []
        with open(anno_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 6:
                    continue
                x, y, w, h = map(int, parts[:4])
                category = int(parts[5])

                if category == 0 or category == 11:
                    continue  # skip ignored and others

                cls_id = category - 1  # shift 1-10 → 0-9
                bbox = convert_box(img_size, (x, y, w, h))
                yolo_labels.append(f"{cls_id} {' '.join([f'{b:.6f}' for b in bbox])}\n")

        # Write YOLO label file in the appropriate folder
        with open(label_dest, 'w') as out:
            out.writelines(yolo_labels)

# Run it on your datasets
convert_visdrone_to_yolo('datasets/VisDrone/VisDrone2019-DET-train', 'dataset')
convert_visdrone_to_yolo('datasets/VisDrone/VisDrone2019-DET-val', 'dataset')
