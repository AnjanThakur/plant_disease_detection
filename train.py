import torch
import os
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm
import time
from pathlib import Path
from models.CNN import CNN
from scripts.dataset import get_transforms, PlantDataset
from torch.utils.data.sampler import WeightedRandomSampler
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score

class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model, f1):
        score = f1  # Track based on F1 score
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.counter = 0
            # Save the best model
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Best model saved with F1 score: {f1:.4f}")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def train_model(model, train_loader, val_loader, num_epochs=30, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    model.to(device)  # Ensure model is on the correct device
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)  # Define scheduler

    # Load checkpoint if exists
    if os.path.exists('best_model.pth'):
        print("Loading checkpoint...")
        model.load_state_dict(torch.load('best_model.pth', map_location=device))
    else:
        print("No checkpoint found, starting fresh training.")

    total_start_time = time.time()
    early_stopping = EarlyStopping(patience=5)  # Initialize early stopping

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        # Progress bar for batches within epoch
        batch_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False, position=1)

        for inputs, targets in batch_pbar:
            inputs, targets = inputs.to(device), targets.to(device)  # Move data to the correct device
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update batch progress bar
            batch_pbar.set_postfix({'loss': f'{train_loss/total:.3f}', 'acc': f'{100.*correct/total:.2f}%'})

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc='Validation', leave=False, position=1)
            for inputs, targets in val_pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

                val_pbar.set_postfix({'val_loss': f'{val_loss/val_total:.3f}', 'val_acc': f'{100.*val_correct/val_total:.2f}%'})

        val_acc = 100. * val_correct / val_total

        # Calculate precision, recall, and F1 score
        precision = precision_score(all_targets, all_preds, average='weighted')
        recall = recall_score(all_targets, all_preds, average='weighted')
        f1 = f1_score(all_targets, all_preds, average='weighted')

        print(f'Validation Precision: {precision:.4f} - Recall: {recall:.4f} - F1 Score: {f1:.4f}')

        # Update learning rate scheduler
        scheduler.step(f1)

        # Early stopping check
        early_stopping(val_loss, model, f1)  # Trigger early stopping based on F1 score
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break  # Stop the training loop if early stopping is triggered

        # Print and update progress
        epoch_time = time.time() - epoch_start_time
        print(f'Epoch {epoch+1}/{num_epochs} - Loss: {train_loss/total:.4f} - Val Acc: {val_acc:.2f}% - Time: {epoch_time:.2f}s')

    total_time = time.time() - total_start_time
    print(f'Training completed in {total_time/60:.2f} minutes')


def main():
    # Paths
    base_path = Path('data')
    train_path = base_path / 'train'
    test_path = base_path / 'test'

    print("Training on CPU")
    device = 'cpu'

    # Transforms
    train_transform, val_transform = get_transforms()

    # Datasets and loaders
    train_dataset = PlantDataset(train_path, transform=train_transform)
    test_dataset = PlantDataset(test_path, transform=val_transform)

    labels = [label for _, label in train_dataset]
    class_counts = Counter(labels)
    class_weights = [1.0 / class_counts[label] for label in labels]
    sampler = WeightedRandomSampler(class_weights, len(class_weights))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, sampler=sampler, num_workers=2, pin_memory=False
    )

    val_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=False
    )

    # Initialize model
    num_classes = len(train_dataset.classes)
    model = CNN(num_classes=num_classes)

    # Train model
    train_model(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
