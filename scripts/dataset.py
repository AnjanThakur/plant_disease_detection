import os
import logging
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms
import cv2
import torch
import random

# Initialize logging
logging.basicConfig(level=logging.ERROR)

class PlantDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load all image paths
        for class_dir in self.root_dir.iterdir():
            if class_dir.is_dir():
                for img_path in class_dir.glob('*.[jp][pn][g]'):
                    try:
                        # Check if the image is valid by loading it
                        with Image.open(img_path) as img:
                            img.verify()  # Verify that the image is not corrupted
                            self.samples.append((str(img_path), self.class_to_idx[class_dir.name]))
                    except Exception as e:
                        logging.error(f"Error loading image {img_path}: {str(e)}")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')  # Convert to RGB
                if self.transform:
                    img = self.transform(img)
                return img, label
        except Exception as e:
            logging.error(f"Error loading image {img_path}: {str(e)}")
            return self.__getitem__((idx + 1) % len(self))

class CannyEdgeDetection:
    def __call__(self, img):
        img = np.array(img)  # Convert PIL image to numpy array

        if img.ndim != 3:  # Check if the image has 3 channels
            logging.error(f"Image has invalid number of channels: {img.shape}")
            return img  # Return the original image if it has an invalid number of channels

        # Convert the image to 8-bit depth before applying Canny edge detection
        if img.dtype != np.uint8:
            img = np.uint8(img)  # Convert to 8-bit if it's not

        edges = cv2.Canny(img, threshold1=100, threshold2=200)  # Apply Canny edge detection
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)  # Convert edges to 3-channel RGB
        return Image.fromarray(edges)
    
def get_transforms():
    train_transform = transforms.Compose([
        CannyEdgeDetection(),  # Apply Canny edge detection
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        CannyEdgeDetection(),  # Apply Canny edge detection
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform

def mixup_data(x, y, alpha=1.0):
    """Applies Mixup to the input data"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).cuda()
        mixed_x = lam * x + (1 - lam) * x[index, :]
        mixed_y = lam * y + (1 - lam) * y[index]
        return mixed_x, mixed_y, lam
    else:
        return x, y, 1

def cutmix_data(x, y, alpha=1.0):
    """Applies CutMix to the input data"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).cuda()
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        y_a, y_b = y, y[index]
        return x, y_a, y_b, lam
    else:
        return x, y, y, 1

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # Uniformly sample the top left corner of the bounding box
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
