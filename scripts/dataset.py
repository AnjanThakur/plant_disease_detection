import os
import logging
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms
import cv2
import torch
import random
import argparse


# Initialize logging with more detail
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_dataset_structure(root_dir):
    """
    Analyzes the structure of a dataset directory to verify class folders and image files.
    """
    root = Path(root_dir)
    
    if not root.exists():
        logging.error(f"Dataset directory does not exist: {root}")
        return [], {}
    
    logging.info(f"Analyzing dataset directory: {root}")
    
    # Get all subdirectories (class folders)
    class_dirs = [d for d in root.iterdir() if d.is_dir()]
    logging.info(f"Found {len(class_dirs)} potential class directories")
    
    if len(class_dirs) == 0:
        logging.error("No class directories found!")
        return [], {}
    
    # Analyze each class directory
    valid_classes = []
    class_images = {}
    valid_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    
    for class_dir in sorted(class_dirs):
        class_name = class_dir.name
        logging.info(f"\nAnalyzing class: {class_name}")
        
        # Get all image files with expanded extensions
        image_files = []
        for ext in valid_extensions:
            image_files.extend(list(class_dir.glob(f"*{ext}")))
        
        # Check for empty directories
        if len(image_files) == 0:
            logging.warning(f"No valid images found in class directory: {class_name}")
            continue
        
        # Verify images can be opened
        valid_images = 0
        invalid_images = []
        
        for img_path in image_files:
            try:
                with Image.open(img_path) as img:
                    img.verify()  # Verify the file is valid
                    valid_images += 1
            except Exception as e:
                invalid_images.append((img_path, str(e)))
        
        logging.info(f"  Total images found: {len(image_files)}")
        logging.info(f"  Valid images: {valid_images}")
        logging.info(f"  Invalid images: {len(invalid_images)}")
        
        if valid_images > 0:
            valid_classes.append(class_name)
            class_images[class_name] = valid_images
            
            if invalid_images:
                for path, error in invalid_images[:5]:  # Show first 5 errors
                    logging.warning(f"    - {path.name}: {error}")
                if len(invalid_images) > 5:
                    logging.warning(f"    ... and {len(invalid_images) - 5} more")
        else:
            logging.error(f"  No valid images in class: {class_name}")
    
    # Summary
    logging.info("\n====== Dataset Summary ======")
    logging.info(f"Total class directories found: {len(class_dirs)}")
    logging.info(f"Valid classes with images: {len(valid_classes)}")
    
    if len(valid_classes) < len(class_dirs):
        logging.warning("Some class directories have no valid images:")
        for d in class_dirs:
            if d.name not in valid_classes:
                logging.warning(f"  - {d.name}")
    
    if valid_classes:
        logging.info("\nClass distribution:")
        total_images = sum(class_images.values())
        for class_name, count in sorted(class_images.items(), key=lambda x: x[1], reverse=True):
            logging.info(f"  {class_name}: {count} images ({count/total_images*100:.1f}%)")
    
    return valid_classes, class_images

def compare_datasets(train_dir, test_dir):
    """
    Compares train and test datasets to ensure class consistency.
    """
    logging.info("\n====== Comparing Train and Test Datasets ======")
    
    train_classes, train_images = analyze_dataset_structure(train_dir)
    test_classes, test_images = analyze_dataset_structure(test_dir)
    
    if not train_classes or not test_classes:
        return
    
    # Find missing classes
    train_set = set(train_classes)
    test_set = set(test_classes)
    
    missing_in_test = train_set - test_set
    missing_in_train = test_set - train_set
    common_classes = train_set.intersection(test_set)
    
    logging.info(f"\nCommon classes: {len(common_classes)}")
    logging.info(f"Classes in train but not in test: {len(missing_in_test)}")
    if missing_in_test:
        for cls in sorted(missing_in_test):
            logging.warning(f"  - {cls}: {train_images.get(cls, 0)} train images, 0 test images")
    
    logging.info(f"Classes in test but not in train: {len(missing_in_train)}")
    if missing_in_train:
        for cls in sorted(missing_in_train):
            logging.warning(f"  - {cls}: 0 train images, {test_images.get(cls, 0)} test images")
    
    # Analyze class distribution ratio
    logging.info("\nTrain/Test Split Analysis:")
    for cls in sorted(common_classes):
        train_count = train_images.get(cls, 0)
        test_count = test_images.get(cls, 0)
        total = train_count + test_count
        train_ratio = train_count / total * 100 if total > 0 else 0
        test_ratio = test_count / total * 100 if total > 0 else 0
        
        logging.info(f"  {cls}: {train_count} train ({train_ratio:.1f}%), {test_count} test ({test_ratio:.1f}%)")

class PlantDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, force_all_classes=False, class_list=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.force_all_classes = force_all_classes
        self.predefined_classes = class_list
        self.class_sample_counts = {}
        self.valid_images_by_class = {}  # Store valid images by class for fallback
        
        # Debug: Print root directory
        logging.info(f"Loading dataset from: {self.root_dir}")
        
        # Get all subdirectories - fixed to ensure all_dirs is always defined
        all_dirs = [d for d in self.root_dir.iterdir() if d.is_dir()]
        logging.info(f"Found {len(all_dirs)} potential class directories")
        
        # If class_list is provided, use it instead of discovering classes
        if self.predefined_classes:
            self.classes = self.predefined_classes
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
            logging.info(f"Using predefined class list with {len(self.classes)} classes")
        else:
            # Discover classes from directories
            self.classes = []
            self.class_to_idx = {}
        
        # Load all image paths and count samples per class
        for class_dir in all_dirs:
            if class_dir.is_dir():
                class_name = class_dir.name
                # Skip classes not in predefined list if we're using one
                if self.predefined_classes and class_name not in self.predefined_classes:
                    logging.info(f"Skipping class '{class_name}' as it's not in the predefined list")
                    continue
                
                # Look for more image formats
                class_images = []
                for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                    class_images.extend(list(class_dir.glob(f"*{ext}")))
                
                total_images = len(class_images)
                logging.info(f"Class '{class_name}': Found {total_images} image files")
                
                valid_images = []
                for img_path in class_images:
                    try:
                        # Check if the image is valid by loading it
                        with Image.open(img_path) as img:
                            img.verify()  # Verify that the image is not corrupted
                            valid_images.append(img_path)
                    except Exception as e:
                        logging.error(f"Error loading image {img_path}: {str(e)}")
                        # Skip the image if there's an error loading it
                
                if not valid_images:
                    logging.warning(f"No valid images found in class directory: {class_name}")
                    if force_all_classes and (not self.predefined_classes or class_name in self.predefined_classes):
                        logging.info(f"Force including class '{class_name}' even with no valid images")
                        if not self.predefined_classes:  # Only add to classes if not predefined
                            self.classes.append(class_name)
                        self.class_sample_counts[class_name] = 0
                    continue
                
                if not self.predefined_classes:  # Only add to classes if not predefined
                    self.classes.append(class_name)
                self.class_sample_counts[class_name] = len(valid_images)
                self.valid_images_by_class[class_name] = [str(path) for path in valid_images]
                logging.info(f"Class '{class_name}': {len(valid_images)} valid images out of {total_images}")
                
                # Only add samples if we have a valid class_to_idx mapping
                if not self.predefined_classes:
                    class_idx = len(self.classes) - 1
                    self.class_to_idx[class_name] = class_idx
                else:
                    class_idx = self.class_to_idx.get(class_name)
                    
                if class_idx is not None:  # Make sure we have a valid index
                    for img_path in valid_images:
                        self.samples.append((str(img_path), class_idx))
        
        # Sort classes for consistency if not predefined
        if not self.predefined_classes:
            self.classes = sorted(self.classes)
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
            
            # Update indices to match sorted classes
            updated_samples = []
            for path, _ in self.samples:
                class_name = Path(path).parent.name
                if class_name in self.class_to_idx:
                    updated_samples.append((path, self.class_to_idx[class_name]))
                else:
                    logging.warning(f"Image {path} belongs to unknown class {class_name}, skipping")
            self.samples = updated_samples
        
        # Debug: Print class distribution
        logging.info(f"Found {len(self.classes)} valid classes with images")
        for cls in self.classes:
            count = self.class_sample_counts.get(cls, 0)
            logging.info(f"Class '{cls}': {count} valid images")
            
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, class_idx = self.samples[idx]
        
        try:
            img = Image.open(img_path).convert('RGB')
            
            if self.transform:
                img = self.transform(img)
                
            return img, class_idx
        except Exception as e:
            logging.error(f"Error loading image at runtime: {img_path}, {str(e)}")
            
            # Get class name from path
            class_name = Path(img_path).parent.name
            
            # Return a random valid image from the same class if available
            if class_name in self.valid_images_by_class and self.valid_images_by_class[class_name]:
                fallback_paths = [p for p in self.valid_images_by_class[class_name] if p != img_path]
                if fallback_paths:
                    try:
                        fallback_path = random.choice(fallback_paths)
                        logging.info(f"Using fallback image from same class: {fallback_path}")
                        img = Image.open(fallback_path).convert('RGB')
                        if self.transform:
                            img = self.transform(img)
                        return img, class_idx
                    except Exception as fallback_e:
                        logging.error(f"Error with fallback image: {fallback_e}")
            
            # If all fails, return a placeholder
            logging.warning(f"Using zero tensor placeholder for class {class_name}")
            placeholder = torch.zeros((3, 224, 224))
            return placeholder, class_idx

class CannyEdgeDetection:
    def __call__(self, img):
        try:
            img = np.array(img)  # Convert PIL image to numpy array

            if img.ndim != 3:  # Check if the image has 3 channels
                logging.error(f"Image has invalid number of channels: {img.shape}")
                # Convert to 3 channels if needed
                if img.ndim == 2:  # Grayscale
                    img = np.stack((img,) * 3, axis=-1)
                else:
                    return Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))  # Return blank image

            # Convert the image to 8-bit depth before applying Canny edge detection
            if img.dtype != np.uint8:
                # Scale to 0-255 range
                img = np.clip(img, 0, 255).astype(np.uint8)

            # Apply Canny edge detection to each channel separately
            edges = np.zeros_like(img)
            for i in range(3):
                edges[:, :, i] = cv2.Canny(img[:, :, i], threshold1=100, threshold2=200)
            
            return Image.fromarray(edges)
        except Exception as e:
            logging.error(f"Error in Canny edge detection: {str(e)}")
            # Return blank image in case of error
            return Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    
def get_transforms(use_canny=False):  # Changed default to False to avoid potential issues
    """
    Get transforms for training and validation.
    Args:
        use_canny: If True, apply Canny edge detection.
    """
    if use_canny:
        edge_transform = CannyEdgeDetection()
    else:
        edge_transform = transforms.Lambda(lambda x: x)  # Identity transform

    train_transform = transforms.Compose([
        edge_transform,  # Apply Canny edge detection if enabled
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        edge_transform,  # Apply Canny edge detection if enabled
        transforms.Resize(256),  # Resize to slightly larger than crop size
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform

def get_all_classes(train_dir, test_dir):
    """
    Get a unified list of all classes from both train and test directories.
    """
    train_classes, _ = analyze_dataset_structure(train_dir)
    test_classes, _ = analyze_dataset_structure(test_dir)
    
    # Combine classes from both directories
    all_classes = sorted(set(train_classes).union(set(test_classes)))
    logging.info(f"Combined list of classes: {len(all_classes)} classes")
    return all_classes

def main():
    parser = argparse.ArgumentParser(description='Analyze dataset structure for deep learning')
    parser.add_argument('--train', type=str, default='data/train', help='Path to training dataset')
    parser.add_argument('--test', type=str, default='data/test', help='Path to test dataset')
    args = parser.parse_args()
    
    compare_datasets(args.train, args.test)

if __name__ == "__main__":
    main()