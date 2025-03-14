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
import logging
from PIL import Image
import matplotlib.pyplot as plt
import random

# Custom imports (ensure these modules exist in your project)
from models.CNN import CNN  # Your CNN model
from scripts.dataset import PlantDataset, get_transforms, get_all_classes  # Your dataset and transforms

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

# Improved Mixup Data Function
def mixup_data(x, y, alpha=1.0, device='cpu'):
    """
    Implements mixup: https://arxiv.org/abs/1710.09412
    Returns mixed inputs, pairs of targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    
    # Create a randomly permuted index
    index = torch.randperm(batch_size).to(device)
    
    # Create mixed samples
    mixed_x = lam * x + (1 - lam) * x[index, :]
    
    # Return mixed inputs, pairs of targets, and mixing coefficient
    return mixed_x, y, y[index], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Calculate loss using mixup principle
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Debugging function to log sample images
def log_sample_images(dataset, num_samples=5):
    """
    Log sample images from each class for debugging.
    """
    logging.info("\n====== Logging Sample Images ======")
    for class_idx, class_name in enumerate(dataset.classes):
        logging.info(f"\nClass {class_name} (Index {class_idx}):")
        class_samples = [sample for sample in dataset.samples if sample[1] == class_idx]
        if not class_samples:
            logging.warning(f"No samples found for class {class_name}")
            continue
        
        # Randomly sample a subset of images
        sampled_images = random.sample(class_samples, min(num_samples, len(class_samples)))
        
        for i, (img_path, _) in enumerate(sampled_images):
            logging.info(f"Sample {i+1}: {img_path}")
            try:
                # Verify the image can be loaded
                with Image.open(img_path) as img:
                    img.verify()
            except Exception as e:
                logging.error(f"Error loading image {img_path}: {str(e)}")

# Training Function
def train_model(model, train_loader, val_loader, num_epochs=30, class_names=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    model.to(device)
    print(f"Model is on: {next(model.parameters()).device}")
    
    # Save class names for future validation
    if class_names:
        torch.save(class_names, 'class_names.pt')
        print(f"Saved {len(class_names)} class names for later validation")

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
                inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=1.0, device=device)
            else:
                targets_a, targets_b = targets, targets
                lam = 1.0

            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Calculate loss with mixup
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            
            # For accuracy calculation during mixup, we use the original labels
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
        
        # Calculate validation metrics
        val_precision = precision_score(all_targets, all_preds, average='weighted')
        val_recall = recall_score(all_targets, all_preds, average='weighted')
        val_f1 = f1_score(all_targets, all_preds, average='weighted')
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"Train Loss: {train_loss/train_total:.4f}, Train Acc: {100.*train_correct/train_total:.2f}%")
        print(f"Val Loss: {val_loss/val_total:.4f}, Val Acc: {100.*val_correct/val_total:.2f}%")
        print(f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")
        
        # Update learning rate scheduler
        scheduler.step(val_f1)
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.6f}")

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

    # Print directory information
    print(f"Training data path: {train_path}")
    print(f"Testing data path: {test_path}")
    
    if not train_path.exists() or not test_path.exists():
        print("ERROR: Data directories not found!")
        return

    # Get a unified list of all classes from both train and test
    all_classes = get_all_classes(train_path, test_path)
    print(f"Total number of unique classes: {len(all_classes)}")
    
    # Transforms
    train_transform, val_transform = get_transforms()

    # Datasets and loaders
    print("Loading training dataset...")
    train_dataset = PlantDataset(train_path, transform=train_transform, force_all_classes=True, class_list=all_classes)
    print(f"Number of training classes found: {len(train_dataset.classes)}")
    print(f"Classes: {train_dataset.classes}")
    print(f"Training samples: {len(train_dataset)}")
    
    print("\nLoading testing dataset...")
    test_dataset = PlantDataset(test_path, transform=val_transform, force_all_classes=True, class_list=all_classes)
    print(f"Number of testing classes found: {len(test_dataset.classes)}")
    print(f"Testing samples: {len(test_dataset)}")
    
    # Log sample images for debugging
    log_sample_images(train_dataset)
    log_sample_images(test_dataset)
    
    # Check if class names match between train and test
    missing_classes = set(test_dataset.classes) - set(train_dataset.classes)
    if missing_classes:
        print(f"WARNING: Classes in testing but not in training: {missing_classes}")
        print("These classes will be excluded from testing.")
        # Filter out missing classes from the test dataset
        test_dataset.samples = [sample for sample in test_dataset.samples if sample[1] in train_dataset.class_to_idx]
        test_dataset.classes = train_dataset.classes
        test_dataset.class_to_idx = train_dataset.class_to_idx
        print(f"Updated test dataset to exclude missing classes. New test samples: {len(test_dataset)}")

    # Use a unified set of classes
    unified_classes = sorted(set(train_dataset.classes).union(set(test_dataset.classes)))
    print(f"Using unified class list with {len(unified_classes)} classes")
    
    # Handle class imbalance with WeightedRandomSampler
    labels = [label for _, label in train_dataset]
    class_counts = Counter(labels)
    
    print("\nClass distribution in training set:")
    for class_idx, count in class_counts.items():
        class_name = train_dataset.classes[class_idx]
        print(f"Class {class_name}: {count} samples")
    
    # Create weights for sampling that account for all classes
    weights = []
    for _, label in train_dataset:
        if label in class_counts and class_counts[label] > 0:
            class_weight = 1.0 / class_counts[label]
        else:
            # For classes with no samples, use a small weight
            class_weight = 0.0001
        weights.append(class_weight)
    
    sampler = WeightedRandomSampler(weights, len(weights))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, sampler=sampler, num_workers=2, pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
    )

    # Initialize model with the unified class count
    num_classes = len(unified_classes)
    print(f"\nInitializing model with {num_classes} output classes")
    model = CNN(num_classes=num_classes)

    # Train model
    train_model(model, train_loader, val_loader, class_names=unified_classes)

if __name__ == "__main__":
    main()