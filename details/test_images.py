import os
import shutil
from pathlib import Path
import hashlib

# Paths
train_dir = Path("data/Tomato/train_tomato")
val_dir = Path("data/Tomato/valid_tomato")
test_dir = Path("data/Tomato/test_tomato")

# Hashing utility
def calculate_hash(image_path):
    with open(image_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def get_image_hashes(directory):
    hashes = set()
    for path in directory.rglob("*"):
        if path.suffix.lower() in [".jpg", ".jpeg", ".png"] and path.is_file():
            hashes.add(calculate_hash(path))
    return hashes

def move_unique_to_test(num_per_class=100):
    print("🔍 Scanning existing hashes...")
    val_hashes = get_image_hashes(val_dir)
    test_hashes = get_image_hashes(test_dir)
    existing_hashes = val_hashes.union(test_hashes)

    moved_total = 0
    for class_folder in train_dir.iterdir():
        if not class_folder.is_dir():
            continue

        class_name = class_folder.name
        test_class_path = test_dir / class_name
        test_class_path.mkdir(parents=True, exist_ok=True)

        moved = 0
        for image_path in class_folder.glob("*"):
            if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            h = calculate_hash(image_path)
            if h not in existing_hashes:
                shutil.move(str(image_path), str(test_class_path / image_path.name))
                existing_hashes.add(h)
                moved += 1
                moved_total += 1
                if moved >= num_per_class:
                    break

        print(f"✅ Moved {moved} images from train/{class_name} to test/{class_name}")

    print(f"\n🎉 Done! Total images moved: {moved_total}")

if __name__ == "__main__":
    move_unique_to_test(num_per_class=300)
