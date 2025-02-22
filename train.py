import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm
import time
from pathlib import Path
from torch.utils.data.sampler import WeightedRandomSampler
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

# Custom imports (ensure these modules exist in your project)
from models.CNN import CNN  # Your CNN model
from scripts.dataset import PlantDataset, get_transforms  # Your dataset and transforms

# Early Stopping Class
class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_f1, model):
        if self.best_score is None or val_f1 > self.best_score:
            self.best_score = val_f1
            self.counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Best model saved with F1 score: {val_f1:.4f}")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# Mixup Data Function
def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size)  # Shuffle indices

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, torch.tensor(lam, dtype=torch.float32)  # Return lam as a tensor

# Training Function
def train_model(model, train_loader, val_loader, num_epochs=30):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    print(f"GPU available: {torch.cuda.is_available()}")
    print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    model.to(device)
    print(f"Model is on: {next(model.parameters()).device}")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    total_start_time = time.time()
    early_stopping = EarlyStopping(patience=5)

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        batch_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training', leave=False)
        for inputs, targets in batch_pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Apply Mixup
            use_mixup = True  # Enable Mixup
            if use_mixup:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=1.0)
                targets_a, targets_b = targets_a.to(device), targets_b.to(device)
                lam = lam.to(device)  # Move lam to the correct device
            else:
                targets_a, targets_b = targets, targets
                lam = torch.tensor(1.0, dtype=torch.float32).to(device)  # Ensure lam is a tensor on the correct device

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

            # Update progress bar
            batch_pbar.set_postfix({'loss': f'{train_loss/train_total:.3f}', 'acc': f'{100.*train_correct/train_total:.2f}%'})

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            batch_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation', leave=False)
            for inputs, targets in batch_pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                # Update progress bar
                batch_pbar.set_postfix({'loss': f'{val_loss/val_total:.3f}', 'acc': f'{100.*val_correct/val_total:.2f}%'})
        
        val_f1 = f1_score(all_targets, all_preds, average='weighted')
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"Train Loss: {train_loss/train_total:.4f}, Train Acc: {100.*train_correct/train_total:.2f}%")
        print(f"Val Loss: {val_loss/val_total:.4f}, Val Acc: {100.*val_correct/val_total:.2f}%, Val F1: {val_f1:.4f}")
        
        # Update learning rate scheduler
        scheduler.step(val_f1)

        epoch_end_time = time.time()
        print(f"Epoch {epoch+1} time: {epoch_end_time - epoch_start_time:.2f}s")
        
        # Early stopping check
        early_stopping(val_f1, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    total_end_time = time.time()
    print(f"Total training time: {total_end_time - total_start_time:.2f}s")

# Main Function
def main():
    # Paths
    base_path = Path('data')
    train_path = base_path / 'train'
    test_path = base_path / 'test'

    # Transforms
    train_transform, val_transform = get_transforms()

    # Datasets and loaders
    train_dataset = PlantDataset(train_path, transform=train_transform)
    test_dataset = PlantDataset(test_path, transform=val_transform)

    # Handle class imbalance with WeightedRandomSampler
    labels = [label for _, label in train_dataset]
    class_counts = Counter(labels)
    class_weights = 1.0 / torch.tensor(list(class_counts.values()), dtype=torch.float)
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, sampler=sampler, num_workers=2, pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
    )

    # Initialize model
    num_classes = len(train_dataset.classes)
    model = CNN(num_classes=num_classes)

    # Train model
    train_model(model, train_loader, val_loader)

if __name__ == "__main__":
    main()