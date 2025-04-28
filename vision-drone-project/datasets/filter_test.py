import os
from pathlib import Path

# Input and output paths
label_dir = Path('C:/Users/havis/OneDrive/Desktop/vision-drone-project/datasets/pseudo-labeled/labels')  # Labels in pseudo-labeled
output_dir = Path('C:/Users/havis/OneDrive/Desktop/vision-drone-project/datasets/pseudo-labeled/filtered_labels')  # Output filtered labels

# Create the output folder if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

# Desired classes to keep: 3, 7, 8
desired_classes = {'3', '7', '8'}

count = 0

# Process each label file
for label_path in label_dir.glob('*.txt'):
    valid_lines = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            class_id = parts[0]  # The class is the first part of each line
            if class_id in desired_classes:
                valid_lines.append(line)  # Keep the line if it matches a desired class

    if valid_lines:
        # Write the valid lines to a new file in the output folder
        out_label = output_dir / label_path.name
        with open(out_label, 'w') as f:
            f.writelines(valid_lines)
            count += 1

print(f"✅ Done. Filtered {count} label files based on the classes 3, 7, 8.")
