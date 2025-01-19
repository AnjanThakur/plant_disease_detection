import os
import shutil
from sklearn.model_selection import train_test_split

def split_data(original_dir, train_dir, test_dir, test_size=0.2):
    """
    Splits the dataset into train and test sets based on the specified test size.
    
    Args:
        original_dir (str): Path to the dataset containing class folders.
        train_dir (str): Path to save the train dataset.
        test_dir (str): Path to save the test dataset.
        test_size (float): Proportion of the dataset to use as test data.
    """
    # Iterate through each class folder in the original directory
    for class_name in os.listdir(original_dir):
        class_dir = os.path.join(original_dir, class_name)
        if os.path.isdir(class_dir):
            # Create corresponding directories in train and test
            os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

            # Get all image files in the current class folder
            image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            # Split the files into train and test sets
            train_files, test_files = train_test_split(image_files, test_size=test_size, random_state=42)

            # Copy files to the train and test directories
            for file_name in train_files:
                src_path = os.path.join(class_dir, file_name)
                dest_path = os.path.join(train_dir, class_name, file_name)
                shutil.copy(src_path, dest_path)

            for file_name in test_files:
                src_path = os.path.join(class_dir, file_name)
                dest_path = os.path.join(test_dir, class_name, file_name)
                shutil.copy(src_path, dest_path)

            print(f"Class '{class_name}': {len(train_files)} images for training, {len(test_files)} images for testing.")

# Paths to original and target directories
original_dir = "data/PlantVillage"  # Path to the new dataset with 15 classes
train_dir = "data/train"           # Path to save train data
test_dir = "data/test"             # Path to save test data

# Split the dataset (80% train, 20% test)
split_data(original_dir, train_dir, test_dir, test_size=0.2)
