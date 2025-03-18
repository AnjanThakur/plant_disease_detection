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
import matplotlib.pyplot as plt
import random
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)

# Custom imports
from models.CNN import CNN
from scripts.dataset import PlantDataset, get_transforms, get_all_classes

# Early Stopping Class
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_path = 'models/best_model.pth'

    def __call__(self, val_f1, model):
        # First score or improved score
        if self.best_score is None or val_f1 > self.best_score + self.min_delta:
            self.best_score = val_f1
            self.counter = 0
            self.save_checkpoint(val_f1, model)
        # Score didn't improve enough
        else:
            self.counter += 1
            logging.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                logging.info("Early stopping triggered!")
    
    def save_checkpoint(self, val_f1, model):
        """Save model when validation F1 improves."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.best_model_path), exist_ok=True)
        
        torch.save(model.state_dict(), self.best_model_path)
        logging.info(f"Best model saved with F1 score: {val_f1:.4f}")

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

# Training Function
def train_model(model, train_loader, val_loader, num_epochs=30, class_names=None, learning_rate=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Training on {device}")
    logging.info(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"GPU name: {torch.cuda.get_device_name(0)}")
    
    model.to(device)
    logging.info(f"Model is on: {next(model.parameters()).device}")
    
    # Save class names for future validation
    if class_names:
        os.makedirs('models', exist_ok=True)
        torch.save(class_names, 'models/class_names.pt')
        logging.info(f"Saved {len(class_names)} class names for later validation")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # Instead of relying on verbose=True, manually log learning rate
    logging.info(f"Learning rate at start: {scheduler.optimizer.param_groups[0]['lr']}")


    # For tracking metrics
    history = {
        'train_loss': [], 'train_acc': [], 
        'val_loss': [], 'val_acc': [], 
        'val_f1': [], 'val_precision': [], 'val_recall': []
    }

    total_start_time = time.time()
    early_stopping = EarlyStopping(patience=7, min_delta=0.005)

    # Create a directory for checkpoints
    os.makedirs('checkpoints', exist_ok=True)

    # First stage: train only the classifier for a few epochs
    model.freeze_feature_extractor()
    logging.info("Stage 1: Training only the classifier (feature extractor frozen)")
    
    for epoch in range(min(5, num_epochs)):  # First 5 epochs or fewer
        _train_epoch(
            epoch, model, train_loader, val_loader, criterion, optimizer, 
            scheduler, device, history, early_stopping, num_epochs, 
            stage="Classifier Only"
        )
        
        if early_stopping.early_stop:
            break
    
    # Second stage: train the full model
    model.unfreeze_feature_extractor()
    # Reduce learning rate for fine-tuning
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate / 10
    
    logging.info("Stage 2: Fine-tuning the entire model (feature extractor unfrozen)")
    
    # Reset early stopping for the second phase
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)
    
    for epoch in range(min(5, num_epochs), num_epochs):
        _train_epoch(
            epoch, model, train_loader, val_loader, criterion, optimizer, 
            scheduler, device, history, early_stopping, num_epochs,
            stage="Full Model"
        )
        
        if early_stopping.early_stop:
            break
            
    total_end_time = time.time()
    logging.info(f"Total training time: {total_end_time - total_start_time:.2f}s")
    
    # Plot training history
    _plot_training_history(history)
    
    # Load best model for final evaluation
    try:
        model.load_state_dict(torch.load('models/best_model.pth'))
        logging.info("Loaded best model for final evaluation")
    except Exception as e:
        logging.error(f"Could not load best model: {e}")
    
    # Final evaluation
    logging.info("Performing final evaluation...")
    model.eval()
    val_loss, val_acc, val_precision, val_recall, val_f1 = _evaluate_model(model, val_loader, criterion, device)
    
    logging.info("\n" + "="*50)
    logging.info("FINAL MODEL PERFORMANCE:")
    logging.info(f"Validation Accuracy: {val_acc:.4f}")
    logging.info(f"Validation F1 Score: {val_f1:.4f}")
    logging.info(f"Validation Precision: {val_precision:.4f}")
    logging.info(f"Validation Recall: {val_recall:.4f}")
    logging.info("="*50 + "\n")
    
    return model, history

def _train_epoch(epoch, model, train_loader, val_loader, criterion, optimizer, scheduler, 
                device, history, early_stopping, num_epochs, stage="Training"):
    """Execute one epoch of training and validation"""
    epoch_start_time = time.time()
    
    # Training phase
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    batch_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - {stage}', leave=False)
    for inputs, targets in batch_pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Apply Mixup with probability 0.5
        use_mixup = random.random() > 0.5
        if use_mixup:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=0.2, device=device)
        else:
            targets_a, targets_b = targets, targets
            lam = 1.0

        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Calculate loss with mixup
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        train_loss += loss.item() * targets.size(0)
        
        # For accuracy calculation during mixup, we use the original labels
        _, predicted = outputs.max(1)
        train_total += targets.size(0)
        train_correct += predicted.eq(targets).sum().item()

        # Update progress bar
        batch_pbar.set_postfix({
            'loss': f'{train_loss/train_total:.3f}', 
            'acc': f'{100.*train_correct/train_total:.2f}%'
        })

    # Calculate epoch metrics
    train_loss = train_loss / train_total
    train_acc = train_correct / train_total
    
    # Validation phase
    val_loss, val_acc, val_precision, val_recall, val_f1 = _evaluate_model(
        model, val_loader, criterion, device
    )
    
    # Update history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['val_f1'].append(val_f1)
    history['val_precision'].append(val_precision)
    history['val_recall'].append(val_recall)
    
    # Print epoch results
    epoch_end_time = time.time()
    logging.info(f"Epoch {epoch+1}/{num_epochs} ({stage}):")
    logging.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {100.*train_acc:.2f}%")
    logging.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {100.*val_acc:.2f}%")
    logging.info(f"  Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")
    logging.info(f"  Time: {epoch_end_time - epoch_start_time:.2f}s")
    
    # Update learning rate scheduler
    scheduler.step(val_f1)
    logging.info(f"  Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Save checkpoint every 5 epochs
    if (epoch + 1) % 5 == 0:
        checkpoint_path = f'checkpoints/model_epoch_{epoch+1}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_f1': val_f1
        }, checkpoint_path)
        logging.info(f"  Checkpoint saved to {checkpoint_path}")
    
    # Early stopping check
    early_stopping(val_f1, model)
    return val_f1

def _evaluate_model(model, dataloader, criterion, device):
    """Evaluate model performance on a dataset"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    val_loss = running_loss / total
    val_acc = correct / total
    
    # Calculate metrics
    val_precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    val_recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    val_f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    
    return val_loss, val_acc, val_precision, val_recall, val_f1

def _plot_training_history(history):
    """Plot training and validation metrics"""
    # Create directory for figures
    os.makedirs('figures', exist_ok=True)
    
    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('figures/loss_history.png')
    
    # Plot accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('figures/accuracy_history.png')
    
    # Plot validation metrics
    plt.figure(figsize=(10, 5))
    plt.plot(history['val_f1'], label='F1 Score')
    plt.plot(history['val_precision'], label='Precision')
    plt.plot(history['val_recall'], label='Recall')
    plt.title('Validation Metrics over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig('figures/validation_metrics.png')
    
    logging.info("Training history plots saved to 'figures/' directory")

# Main Function
def main():
    # Paths
    base_path = Path('data')
    train_path = base_path / 'train'
    test_path = base_path / 'test'

    # Print directory information
    logging.info(f"Training data path: {train_path}")
    logging.info(f"Testing data path: {test_path}")
    
    if not train_path.exists() or not test_path.exists():
        logging.error("ERROR: Data directories not found!")
        return

    # Get a unified list of all classes from both train and test
    all_classes = get_all_classes(train_path, test_path)
    logging.info(f"Total number of unique classes: {len(all_classes)}")
    
    # Transforms
    train_transform, val_transform = get_transforms()

    # Datasets and loaders
    logging.info("Loading training dataset...")
    train_dataset = PlantDataset(train_path, transform=train_transform, force_all_classes=True, class_list=all_classes)
    logging.info(f"Number of training classes found: {len(train_dataset.classes)}")
    logging.info(f"Training samples: {len(train_dataset)}")
    
    logging.info("\nLoading testing dataset...")
    test_dataset = PlantDataset(test_path, transform=val_transform, force_all_classes=True, class_list=all_classes)
    logging.info(f"Number of testing classes found: {len(test_dataset.classes)}")
    logging.info(f"Testing samples: {len(test_dataset)}")
    
    # Check for class consistency
    missing_classes = set(test_dataset.classes) - set(train_dataset.classes)
    if missing_classes:
        logging.warning(f"WARNING: Classes in testing but not in training: {missing_classes}")
        logging.warning("These classes will be excluded from testing.")
        # Filter out missing classes from the test dataset
        test_dataset.samples = [sample for sample in test_dataset.samples 
                               if sample[1] in set(train_dataset.class_to_idx.values())]
        logging.info(f"Updated test dataset to exclude missing classes. New test samples: {len(test_dataset)}")

    # Use a unified set of classes
    unified_classes = sorted(set(train_dataset.classes).union(set(test_dataset.classes)))
    logging.info(f"Using unified class list with {len(unified_classes)} classes")
    
    # Handle class imbalance with WeightedRandomSampler
    labels = [label for _, label in train_dataset]
    class_counts = Counter(labels)
    
    # Log class distribution
    logging.info("\nClass distribution in training set:")
    for class_idx, count in sorted(class_counts.items()):
        if class_idx < len(train_dataset.classes):  # Safety check
            class_name = train_dataset.classes[class_idx]
            logging.info(f"Class {class_name}: {count} samples")
    
    # Create weights for sampling to handle class imbalance
    weights = []
    for _, label in train_dataset:
        if label in class_counts and class_counts[label] > 0:
            class_weight = 1.0 / class_counts[label]
        else:
            # For classes with no samples, use a small weight
            class_weight = 0.0001
        weights.append(class_weight)
    
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    # Set batch size based on available memory
    batch_size = 16 if torch.cuda.is_available() else 8
    logging.info(f"Using batch size of {batch_size}")

    train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    sampler=sampler, 
    num_workers=0,  # Set to 0 for debugging. Increase later if needed.
    pin_memory=torch.cuda.is_available()
)

    val_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    # Initialize model with the unified class count
    num_classes = len(unified_classes)
    logging.info(f"\nInitializing model with {num_classes} output classes")
    
    # Create model
    model = CNN(num_classes=num_classes)
    
    # Train model
    train_model(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        num_epochs=30,  # Adjust as needed
        class_names=unified_classes, 
        learning_rate=3e-4  # Adjust as needed
    )

if __name__ == "__main__":
    main()