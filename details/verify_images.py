import os
import hashlib
from pathlib import Path
from collections import defaultdict
import shutil

# Set your dataset directories
train_dir = Path("data/Tomato/train_tomato")
val_dir = Path("data/Tomato/valid_tomato")
test_dir = Path("data/Tomato/test_tomato")

# Prioritization for keeping images
priority_order = [train_dir, val_dir, test_dir]

def get_all_images(base_dir):
    return [p for p in base_dir.rglob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]

def calculate_hash(image_path):
    with open(image_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def remove_duplicates():
    hash_map = defaultdict(list)

    # Step 1: Calculate hash for every image
    for dataset in priority_order:
        for image_path in get_all_images(dataset):
            img_hash = calculate_hash(image_path)
            hash_map[img_hash].append(image_path)

    # Step 2: Remove duplicates based on priority
    for img_hash, paths in hash_map.items():
        if len(paths) > 1:
            # Keep the image from the highest-priority dataset
            paths.sort(key=lambda p: priority_order.index([d for d in priority_order if d in p.parents][0]))
            keep_path = paths[0]
            for duplicate_path in paths[1:]:
                print(f"🗑️ Removing duplicate: {duplicate_path}")
                os.remove(duplicate_path)

    print("✅ Duplicate removal complete! Each image now exists in only one dataset.")

if __name__ == "__main__":
    remove_duplicates()
