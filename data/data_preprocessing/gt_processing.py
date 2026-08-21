# -*- coding: utf-8 -*-
import os
from pathlib import Path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # folder this .py file lives in
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))  # up two levels, out of data/data_preprocessing/

# paths
main_gt_file = os.path.join(REPO_ROOT, "Urdu_Word_Dataset", "train", "train_gt.txt")
output_dir = os.path.join(REPO_ROOT, "Urdu_Word_Dataset", "train", "gt_txt")
root_images_dir = os.path.join(REPO_ROOT, "Urdu_Word_Dataset", "train", "images")

os.makedirs(output_dir, exist_ok=True)

with open(main_gt_file, "r", encoding="utf-8") as f:
    for line_no, line in enumerate(f, start=1):

        line = line.strip()
        if not line:
            continue

        image_rel = None
        text = None

        # Case 1: tab-separated (correct)
        if "\t" in line:
            image_rel, text = line.split("\t", 1)

        # Case 2: space-separated (broken lines)
        else:
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                image_rel, text = parts
            else:
                print(f"[WARNING] Skipping malformed line {line_no}: {repr(line)}")
                continue

        # ---- FIX PATH ISSUE HERE ----
        # Remove leading "images/" if present
        if image_rel.startswith("images/"):
            image_rel = image_rel[len("images/"):]

        # Build correct absolute path
        image_path = os.path.join(root_images_dir, image_rel)

        print(f"image path is {image_path} and text is {text}")

        if not os.path.exists(image_path):
            print(f"[WARNING] Image not found (line {line_no}): {image_path}")
            continue

        image_name = os.path.splitext(os.path.basename(image_path))[0]
        out_txt_path = os.path.join(output_dir, f"{image_name}.txt")

        with open(out_txt_path, "w", encoding="utf-8") as out_f:
            out_f.write(text)



folder_path=Path(output_dir)
jpg_files = list(folder_path.glob("*.txt"))
print(f"Number of .txt files in gt_txt folder is: {len(jpg_files)}")
