import os
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch


class PlantDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, transform=None, class_list=None):
        self.data_path = data_path
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = []

        # Create label map dynamically or from class_list
        if class_list:
            self.classes = class_list
        else:
            self.classes = sorted([d.name for d in os.scandir(data_path) if d.is_dir()])

        self.label_map = {class_name: idx for idx, class_name in enumerate(self.classes)}

        for class_folder in self.classes:
            class_folder_path = os.path.join(data_path, class_folder)
            for img_name in os.listdir(class_folder_path):
                img_path = os.path.join(class_folder_path, img_name)
                self.image_paths.append(img_path)
                self.labels.append(class_folder)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = self.label_map[self.labels[idx]]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label

def get_transforms():
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Rotate(limit=20, p=0.5),
        A.RandomResizedCrop(224, 224, scale=(0.8, 1.0), p=1),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    return train_transform, val_transform

def get_all_classes(train_path, test_path):
    all_classes = sorted([d.name for d in os.scandir(train_path) if d.is_dir()])
    return all_classes
