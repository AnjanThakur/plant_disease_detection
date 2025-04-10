import os
import shutil
import random

# Define paths
test_dir = "data/test"  # Change this to your test dataset path
val_dir = "data/validate"  # Change this to your desired validation dataset path

# Create validation directory if it doesn't exist
os.makedirs(val_dir, exist_ok=True)

# Get all class folders
classes = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]

for class_name in classes:
    class_test_path = os.path.join(test_dir, class_name)
    class_val_path = os.path.join(val_dir, class_name)
    
    # Create class directory in validation set
    os.makedirs(class_val_path, exist_ok=True)
    
    # Get all images in the test class folder
    images = [f for f in os.listdir(class_test_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    
    # Shuffle images for randomness
    random.shuffle(images)
    
    # Calculate 10% for validation
    num_val = max(1, int(0.1 * len(images)))  # Ensure at least 1 image per class
    val_images = images[:num_val]
    
    # Move images to validation folder
    for img in val_images:
        src = os.path.join(class_test_path, img)
        dst = os.path.join(class_val_path, img)
        shutil.move(src, dst)
    
    print(f"✅ Moved {num_val} images from '{class_name}' to validation.")

print("✅ Validation dataset created successfully!")
