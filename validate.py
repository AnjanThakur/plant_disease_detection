import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from scripts.dataset import PlantDataset, get_transforms
from models.CNN import CNN
import numpy as np
import logging
import os

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def validate_model(test_loader, model, class_names):
    """
    Validate the trained model on the test dataset and compute metrics.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Validating on {device}")
    
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    all_preds = []
    all_labels = []
    
    # Keep track of class-wise predictions
    class_correct = {i: 0 for i in range(len(class_names))}
    class_total = {i: 0 for i in range(len(class_names))}

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update class-wise accuracy tracking
            for i in range(len(labels)):
                label = labels[i].item()
                pred = predicted[i].item()
                class_total[label] = class_total.get(label, 0) + 1
                if label == pred:
                    class_correct[label] = class_correct.get(label, 0) + 1

    # Compute overall accuracy
    accuracy = sum(class_correct.values()) / sum(class_total.values())
    logging.info(f"Overall Accuracy: {accuracy * 100:.2f}%")
    
    # Print class-wise accuracy
    logging.info("\nClass-wise Accuracy:")
    for i in range(len(class_names)):
        if class_total.get(i, 0) > 0:
            accuracy = class_correct.get(i, 0) / class_total.get(i, 0)
            logging.info(f"{class_names[i]}: {accuracy * 100:.2f}% ({class_correct.get(i, 0)}/{class_total.get(i, 0)})")
        else:
            logging.warning(f"{class_names[i]}: No test samples")

    # Compute metrics
    logging.info("\nClassification Report:")
    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    logging.info(f"\n{report}")

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    logging.info("\nConfusion Matrix:")
    logging.info(f"\n{cm}")

    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    
    # If there are many classes, use a smaller font
    if len(class_names) > 20:
        sns.heatmap(cm, annot=True, fmt='d', cmap="YlGnBu", 
                    xticklabels=[f"C{i}" for i in range(len(class_names))], 
                    yticklabels=[f"C{i}" for i in range(len(class_names))],
                    annot_kws={"size": 8})
        plt.xlabel("Predicted (see class mapping in logs)")
        plt.ylabel("True (see class mapping in logs)")
        # Print class mapping
        logging.info("\nClass mapping for confusion matrix:")
        for i, name in enumerate(class_names):
            logging.info(f"C{i} = {name}")
    else:
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap="YlGnBu")
        plt.xlabel("Predicted")
        plt.ylabel("True")
    
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    logging.info("Confusion matrix saved to 'confusion_matrix.png'")
    
    try:
        plt.show()
    except Exception as e:
        logging.warning(f"Could not display plot: {e}")

def main():
    # Paths
    test_path = 'data/test'

    # Check if test directory exists
    if not os.path.exists(test_path):
        logging.error(f"Test directory not found: {test_path}")
        return

    logging.info(f"Validating model on test data from: {test_path}")

    # Get transforms
    _, val_transform = get_transforms()  # Ensure validation transform matches

    # Load test dataset
    test_dataset = PlantDataset(test_path, transform=val_transform)
    logging.info(f"Number of test classes: {len(test_dataset.classes)}")
    logging.info(f"Test classes: {test_dataset.classes}")
    logging.info(f"Number of test samples: {len(test_dataset)}")
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,  # Match batch size used during training
        shuffle=False,
        num_workers=4
    )
    # Load class names
    if os.path.exists('class_names.pt'):
        class_names = torch.load('class_names.pt')
        logging.info(f"Class names loaded from 'class_names.pt': {class_names}")
    else:
        logging.warning("'class_names.pt' not found. Loading class names from the dataset.")
        class_names = test_dataset.classes
        logging.info(f"Class names loaded from dataset: {class_names}")

    # Check if model file exists
    model_path = 'best_model.pth'
    if not os.path.exists(model_path):
        logging.error(f"Model file not found: {model_path}")
        return

    # Load the trained model
    model = CNN(num_classes=len(class_names))
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        logging.info("\nModel loaded successfully!")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        # Try to load model with different number of classes
        logging.info("Attempting to load model with different class configuration...")
        
        # Check the model's structure to determine the number of classes in the saved model
        try:
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            output_layer_weight = state_dict['classifier.3.weight']
            saved_num_classes = output_layer_weight.size(0)
            logging.info(f"Saved model has {saved_num_classes} output classes, but dataset has {len(class_names)} classes")
            
            if saved_num_classes != len(class_names):
                logging.warning("Class mismatch between model and dataset!")
                
                # Two options:
                # 1. Create a model with the same number of classes as the saved model (for evaluation)
                model = CNN(num_classes=saved_num_classes)
                model.load_state_dict(state_dict)
                logging.info(f"Model loaded with {saved_num_classes} classes instead of {len(class_names)}")
                
                # Update class_names to match the model's output size
                if saved_num_classes < len(class_names):
                    class_names = class_names[:saved_num_classes]
                    logging.warning(f"Truncated class list to first {saved_num_classes} classes")
                else:
                    # Pad class names with "Unknown" if model has more classes than dataset
                    additional_classes = saved_num_classes - len(class_names)
                    class_names = list(class_names) + [f"Unknown_{i}" for i in range(additional_classes)]
                    logging.warning(f"Added {additional_classes} unknown class names")
        except Exception as nested_e:
            logging.error(f"Failed to analyze or load model: {nested_e}")
            return

    # Validate the model
    validate_model(test_loader, model, class_names)

if __name__ == "__main__":
    main()