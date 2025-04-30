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
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
)

# Custom imports
from models.CNN import CNN
from scripts.dataset import PlantDataset, get_transforms

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
# MixUp Function             #
# ========================== #
def mixup(batch_size, inputs, labels, num_classes, alpha=0.4):
    lam = np.random.beta(alpha, alpha)  # Lambda for MixUp
    index = torch.randperm(inputs.size(0)).to(inputs.device)  # Random permutation for mixing
    one_hot_labels = F.one_hot(labels, num_classes=num_classes).float()  # One-hot encode labels

    # Mix the inputs
    mixed_inputs = lam * inputs + (1 - lam) * inputs[index, :]

    # Mix the one-hot encoded labels
    mixed_labels = lam * one_hot_labels + (1 - lam) * one_hot_labels[index]

    return mixed_inputs, mixed_labels

# ========================== #
# Evaluation Function        #
# ========================== #
def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch[:2]
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, F.one_hot(labels, num_classes=outputs.shape[1]).float())
            running_loss += loss.item() * inputs.size(0)

            preds = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

            all_preds.extend(preds.cpu().numpy())
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
# Soft Target CrossEntropy   #
# ========================== #
class SoftTargetCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, soft_targets):
        log_probs = F.log_softmax(logits, dim=1)
        return -(soft_targets * log_probs).sum(dim=1).mean()


# ========================== #
# Training Function          #
# ========================== #
def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=3e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    num_classes = model.features.fc[-1].out_features  # Adapted to your CNN definition
    criterion = SoftTargetCrossEntropyLoss()  # For mixed soft targets
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    # Access the current learning rate like this:
    current_lr = scheduler.get_last_lr()
    print(current_lr)


    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")

        for batch in loop:
            inputs, labels = batch[:2]
            inputs, labels = inputs.to(device), labels.to(device)

            # Apply MixUp (returns mixed inputs and soft targets)
            inputs, labels = mixup(inputs.size(0), inputs, labels, num_classes=num_classes)

            optimizer.zero_grad()
            outputs = model(inputs)

            # Compute loss using soft labels (like CrossEntropy with soft targets)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            # Predictions from model
            preds = torch.argmax(outputs, dim=1)

            # Convert one-hot labels back to class indices
            hard_labels = torch.argmax(labels, dim=1)

            correct += preds.eq(hard_labels).sum().item()
            total += labels.size(0)  # or use hard_labels.size(0)

        # ✅ Now compute average loss and accuracy safely
        if total > 0:
            train_loss = running_loss / total
            train_acc = correct / total
        else:
            train_loss = 0.0
            train_acc = 0.0
            print("⚠️ No training samples were processed. Check the DataLoader.")


        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate_model(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        scheduler.step(val_f1)
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"Current learning rate: {current_lr:.6f}")
        logging.info(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                     f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        early_stopping(val_f1, model)
        if early_stopping.early_stop:
            logging.info("Stopping early due to no improvement.")
            break

    torch.save(model.state_dict(), "models/best_tomato_model.pth")


# ========================== #
# Main Function              #
# ========================== #
def main():
    data_dir = "data/Tomato/train_tomato"
    val_dir = "data/Tomato/valid_tomato"
    os.makedirs("outputs", exist_ok=True)

    train_transform, val_transform = get_transforms()
    train_dataset = PlantDataset(data_dir, transform=train_transform)
    val_dataset = PlantDataset(val_dir, transform=val_transform)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = len(train_dataset.classes)
    model = CNN(num_classes=num_classes).to(device)

    train_model(model, train_loader, val_loader, num_epochs=20, learning_rate=3e-4)

if __name__ == "__main__":
    main()
