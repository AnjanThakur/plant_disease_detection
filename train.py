import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm.auto import tqdm
import logging
import os
from sklearn.metrics import precision_score, recall_score, f1_score
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

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
# Early Stopping Class       #
# ========================== #
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_path = "models/best_tomato_model.pth"

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
# Evaluation Function        #
# ========================== #
def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 3:
                inputs, labels, _ = batch  # Ignore paths if present
            elif len(batch) == 2:
                inputs, labels = batch
            else:
                raise ValueError(f"Unexpected batch format: {len(batch)} elements")
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    if total == 0:
        logging.error("No valid labels or predictions found during validation.")
        return 0, 0, 0, 0, 0

    val_loss = running_loss / total
    val_acc = correct / total
    val_precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    val_recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return val_loss, val_acc, val_precision, val_recall, val_f1

# ========================== #
# Training Function         #
# ========================== #
def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=3e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")

        for batch in loop:
            inputs, labels = batch[:2]  # Ignore img_path

            
            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)  # Move to GPU if available
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        train_loss = running_loss / total
        train_acc = correct / total

        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate_model(model, val_loader, criterion, device)
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        scheduler.step()
        logging.info(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        
        early_stopping(val_f1, model)
        if early_stopping.early_stop:
            logging.info("Stopping early due to no improvement.")
            break

    torch.save(model.state_dict(), "models/best_tomato_model.pth")

# ========================== #
# Main Function             #
# ========================== #
def main():
    data_dir = "data/Tomato/train_tomato"
    val_dir = "data/Tomato/valid_tomato"
    save_dir = "outputs"
    os.makedirs(save_dir, exist_ok=True)

    train_transform, val_transform = get_transforms()
    train_dataset = PlantDataset(data_dir, transform=train_transform)
    val_dataset = PlantDataset(val_dir, transform=val_transform)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = len(train_dataset.classes)
    model = CNN(num_classes=num_classes).to(device)

    train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=3e-4)

if __name__ == "__main__":
    main()
