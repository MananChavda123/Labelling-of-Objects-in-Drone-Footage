import glob

# Path to your auto-labeled files from detect.py
labels_path = "../datasets/VisDrone2019-DET-val/labels/val_auto/labels/"

# Process each label file
for txt_file in glob.glob(labels_path + "*.txt"):
    with open(txt_file, 'r') as f:
        lines = f.readlines()

    # Adjust class IDs by shifting them (+1) as VisDrone dataset uses 1-based class IDs
    with open(txt_file, 'w') as f:
        for line in lines:
            parts = line.split()  # Split the line into [class_id, x_center, y_center, width, height, confidence]
            class_id = int(parts[0]) + 1  # Shift class ID by +1
            # Write the new line with adjusted class_id
            f.write(f"{class_id} {' '.join(parts[1:])}\n")

print("✅ Class IDs successfully shifted.")
