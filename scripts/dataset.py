import os
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging
class PlantDataset(Dataset):
    def __init__(self, data_path, transform=None, class_list=None):
        self.data_path = data_path
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = class_list if class_list else sorted([d.name for d in os.scandir(data_path) if d.is_dir()])
        self.label_map = {class_name: idx for idx, class_name in enumerate(self.classes)}

        valid_extensions = (".jpg", ".jpeg", ".png")
        for class_folder in self.classes:
            class_folder_path = os.path.join(data_path, class_folder)
            if not os.path.exists(class_folder_path):
                print(f"⚠️ Warning: Class folder not found - {class_folder_path}")
                continue

            for img_name in os.listdir(class_folder_path):
                if img_name.lower().endswith(valid_extensions):
                    img_path = os.path.join(class_folder_path, img_name)
                    if os.path.isfile(img_path):
                        self.image_paths.append(img_path)
                        self.labels.append(self.label_map[class_folder])

        print(f"✅ Loaded {len(self.image_paths)} images from {len(self.classes)} classes.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        if not os.path.exists(img_path):
            logging.warning(f"Skipping missing image: {img_path}")
            return self.__getitem__((idx + 1) % len(self))

        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            logging.warning(f"Skipping corrupt image: {img_path}")
            return self.__getitem__((idx + 1) % len(self))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels[idx]
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        
        return image, label  # Return just image and label

def get_transforms():
    train_transform = A.Compose([
        A.RandomResizedCrop(224, 224, scale=(0.6, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.RandomShadow(p=0.3),
        A.ElasticTransform(alpha=1, sigma=50, p=0.3),
        A.Perspective(scale=(0.05, 0.1), p=0.3),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.5),

        # More robust augmentations
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.MotionBlur(blur_limit=7, p=0.5),
        A.ISONoise(p=0.2),
        A.RandomGamma(gamma_limit=(50, 150), p=0.5),
        
        # Normalize for pre-trained models like ResNet
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    return train_transform, val_transform


