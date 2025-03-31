import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm.auto import tqdm
import time
from pathlib import Path
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import logging
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
)

# Custom imports
from models.CNN import CNN
from scripts.dataset import PlantDataset, get_transforms, get_all_classes


# ========================== #
# Advanced Data Augmentations #
# ========================== #

def get_advanced_transforms():
    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Rotate(limit=20, p=0.5),
            A.Resize(224, 224),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose([A.Resize(224, 224), ToTensorV2()])

    return train_transform, val_transform


# ========================== #
# Early Stopping Class       #
# ========================== #
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_path = "models/best_model_with_arch.pth"

    def __call__(self, val_f1, model):
        if self.best_score is None or val_f1 > self.best_score + self.min_delta:
            self.best_score = val_f1
            self.counter = 0
            self.save_checkpoint(val_f1, model)
        else:
            self.counter += 1
            logging.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                logging.info("Early stopping triggered!")

    def save_checkpoint(self, val_f1, model):
        os.makedirs(os.path.dirname(self.best_model_path), exist_ok=True)
        torch.save({"model_state_dict": model.state_dict()}, self.best_model_path)
        logging.info(f"Best model saved with F1 score: {val_f1:.4f}")


# ========================== #
# Evaluation and Training    #
# ========================== #

def _evaluate_model(model, val_loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.float().to(device)
            labels = labels.long().to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = running_loss / total
    val_acc = correct / total
    val_precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    val_recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return val_loss, val_acc, val_precision, val_recall, val_f1


def _train_epoch(epoch, model, train_loader, val_loader, criterion, optimizer,
                 scheduler, device, history, early_stopping, num_epochs, stage="Full Model"):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] ({stage})")

    for inputs, labels in loop:
        inputs = inputs.float().to(device)
        labels = labels.long().to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / total
    train_acc = correct / total

    val_loss, val_acc, val_precision, val_recall, val_f1 = _evaluate_model(model, val_loader, criterion, device)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["val_f1"].append(val_f1)
    history["val_precision"].append(val_precision)
    history["val_recall"].append(val_recall)

    scheduler.step()
    logging.info(
        f"✅ Epoch [{epoch+1}/{num_epochs}] ({stage}) - "
        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}"
    )
    early_stopping(val_f1, model)


# ========================== #
# Updated Training Function  #
# ========================== #

def train_model(model, train_loader, val_loader, num_epochs=50, class_names=None, learning_rate=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Training on {device}")
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "val_precision": [],
        "val_recall": [],
    }

    early_stopping = EarlyStopping(patience=5, min_delta=0.001)

    for epoch in range(num_epochs):
        _train_epoch(
            epoch,
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            device,
            history,
            early_stopping,
            num_epochs,
        )
        if early_stopping.early_stop:
            logging.info("Stopping early due to no improvement.")
            break

    # Save the model weights after training
    torch.save(model.state_dict(), "models/best_model.pth")



# ========================== #
# Dataset and Main Function  #
# ========================== #

def main():
    base_path = Path("data")
    train_path = base_path / "train"
    test_path = base_path / "test"

    if not train_path.exists() or not test_path.exists():
        logging.error("❌ ERROR: Data directories not found!")
        return

    all_classes = get_all_classes(train_path, test_path)
    train_transform, val_transform = get_transforms()

    train_dataset = PlantDataset(train_path, transform=train_transform, class_list=all_classes)
    val_dataset = PlantDataset(test_path, transform=val_transform, class_list=all_classes)

    batch_size = 16 if torch.cuda.is_available() else 8
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available()
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available()
    )

    model = CNN(num_classes=len(all_classes))

    # Start Training
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=50,
        class_names=all_classes,
        learning_rate=3e-4,
    )

if __name__ == "__main__":
    main()
