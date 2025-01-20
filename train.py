import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm
import time
from pathlib import Path
from models.CNN import CNN
from scripts.dataset import get_transforms, PlantDataset, mixup_data, cutmix_data
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

def train_model(model, train_loader, val_loader, num_epochs=30):
    # Ensure GPU is used if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    model.to(device)  # Move model to the selected device

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    total_start_time = time.time()
    early_stopping = EarlyStopping(patience=5)

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        # Progress bar for batches within epoch
        batch_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False, position=1)

        for inputs, targets in batch_pbar:
            inputs, targets = inputs.to(device), targets.to(device)  # Move data to GPU/CPU

            # Apply Mixup or CutMix
            inputs, targets, lam = mixup_data(inputs, targets, alpha=1.0)
            inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets, alpha=1.0)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update batch progress bar
            batch_pbar.set_postfix({'loss': f'{train_loss/total:.3f}', 'acc': f'{100.*correct/total:.2f}%'})

        # Update learning rate scheduler
        scheduler.step(train_loss)

        epoch_end_time = time.time()
        print(f"Epoch {epoch+1} training time: {epoch_end_time - epoch_start_time:.2f}s")
        
        # Call early stopping
        early_stopping(train_loss, model, correct / total)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    total_end_time = time.time()
    print(f"Total training time: {total_end_time - total_start_time:.2f}s")


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

    labels = [label for _, label in train_dataset]
    class_counts = Counter(labels)
    class_weights = [1.0 / class_counts[label] for label in labels]
    sampler = WeightedRandomSampler(class_weights, len(class_weights))

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