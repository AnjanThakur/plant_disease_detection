import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import logging
import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import argparse

# Initialize logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("validation.log"),
        logging.StreamHandler()
    ]
)

def validate_model(test_loader, model, class_names, device='cpu', save_dir='outputs', 
                   save_predictions=False, confidence_threshold=0.5):
    """
    Validate the trained model on the test dataset and compute metrics.
    
    Args:
        test_loader: DataLoader for test dataset
        model: Trained model
        class_names: List of class names
        device: Device to run validation on ('cpu' or 'cuda')
        save_dir: Directory to save outputs
        save_predictions: Whether to save per-sample predictions
        confidence_threshold: Threshold for prediction confidence reporting
    
    Returns:
        accuracy: Overall model accuracy
    """
    logging.info(f"Validating on {device}")
    
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    all_preds = []
    all_labels = []
    all_confidences = []
    all_paths = []
    
    # Keep track of class-wise predictions
    class_correct = {i: 0 for i in range(len(class_names))}
    class_total = {i: 0 for i in range(len(class_names))}

    # Process batches with a progress bar
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Validating"):
            # Handle different batch formats
            if isinstance(batch, tuple) and len(batch) >= 2:
                images = batch[0].to(device)
                labels = batch[1].to(device)
                
                # Check for paths in the batch (some custom datasets provide this)
                paths = None
                if len(batch) >= 3 and isinstance(batch[2], list):
                    paths = batch[2]
            else:
                logging.error(f"Unexpected batch format: {batch}")
                continue
                
            # Get model outputs
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get predicted class and confidence
            confidence_values, predicted = torch.max(probabilities, 1)
            
            # Extend prediction lists
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidence_values.cpu().numpy())
            
            if paths is not None:
                all_paths.extend(paths)
            
            # Update class-wise accuracy tracking
            for i in range(len(labels)):
                label = labels[i].item()
                pred = predicted[i].item()
                
                # Make sure we don't get KeyError for out-of-range labels
                if label in class_total:
                    class_total[label] = class_total.get(label, 0) + 1
                    if label == pred:
                        class_correct[label] = class_correct.get(label, 0) + 1
                else:
                    logging.warning(f"Found label {label} which is out of range for class_names")

    # Compute overall accuracy
    if sum(class_total.values()) > 0:
        accuracy = sum(class_correct.values()) / sum(class_total.values())
        logging.info(f"Overall Accuracy: {accuracy * 100:.2f}%")
    else:
        logging.error("No valid test samples found!")
        return None
    
    # Print class-wise accuracy
    logging.info("\nClass-wise Accuracy:")
    class_accuracy_data = []
    
    for i in range(len(class_names)):
        if class_total.get(i, 0) > 0:
            accuracy = class_correct.get(i, 0) / class_total.get(i, 0)
            logging.info(f"{class_names[i]}: {accuracy * 100:.2f}% ({class_correct.get(i, 0)}/{class_total.get(i, 0)})")
            
            class_accuracy_data.append({
                'class': class_names[i],
                'accuracy': accuracy * 100,
                'correct': class_correct.get(i, 0),
                'total': class_total.get(i, 0)
            })
        else:
            logging.warning(f"{class_names[i]}: No test samples")
    
    # Save class-wise accuracy to CSV
    accuracy_df = pd.DataFrame(class_accuracy_data)
    accuracy_df.to_csv(f"{save_dir}/class_accuracy.csv", index=False)
    logging.info(f"Class-wise accuracy saved to {save_dir}/class_accuracy.csv")
    
    # Plot class-wise accuracy
    plt.figure(figsize=(15, 10))
    ordered_df = accuracy_df.sort_values('accuracy', ascending=False)
    
    # Adjust font size based on number of classes
    if len(class_names) > 30:
        plt.figure(figsize=(20, 12))
        fontsize = 8
    else:
        fontsize = 10
    
    sns.barplot(x='class', y='accuracy', data=ordered_df)
    plt.xticks(rotation=90, fontsize=fontsize)
    plt.title('Class-wise Accuracy')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/class_accuracy.png")
    
    # Compute and save low confidence correct and incorrect predictions
    if save_predictions and all_paths:
        low_confidence_correct = []
        low_confidence_incorrect = []
        
        for i in range(len(all_labels)):
            if all_confidences[i] < confidence_threshold:
                if all_labels[i] == all_preds[i]:
                    low_confidence_correct.append({
                        'path': all_paths[i] if i < len(all_paths) else 'unknown',
                        'true_class': class_names[all_labels[i]] if all_labels[i] < len(class_names) else f"Unknown-{all_labels[i]}",
                        'pred_class': class_names[all_preds[i]] if all_preds[i] < len(class_names) else f"Unknown-{all_preds[i]}",
                        'confidence': all_confidences[i]
                    })
                else:
                    low_confidence_incorrect.append({
                        'path': all_paths[i] if i < len(all_paths) else 'unknown',
                        'true_class': class_names[all_labels[i]] if all_labels[i] < len(class_names) else f"Unknown-{all_labels[i]}",
                        'pred_class': class_names[all_preds[i]] if all_preds[i] < len(class_names) else f"Unknown-{all_preds[i]}",
                        'confidence': all_confidences[i]
                    })
        
        if low_confidence_correct:
            pd.DataFrame(low_confidence_correct).to_csv(f"{save_dir}/low_confidence_correct.csv", index=False)
            logging.info(f"Low confidence correct predictions saved to {save_dir}/low_confidence_correct.csv")
        
        if low_confidence_incorrect:
            pd.DataFrame(low_confidence_incorrect).to_csv(f"{save_dir}/low_confidence_incorrect.csv", index=False)
            logging.info(f"Low confidence incorrect predictions saved to {save_dir}/low_confidence_incorrect.csv")

    # Compute metrics
    target_names = [str(name) for name in class_names]  # Ensure all class names are strings
    
    logging.info("\nClassification Report:")
    try:
        report = classification_report(all_labels, all_preds, target_names=target_names, zero_division=0)
        logging.info(f"\n{report}")
        
        # Save classification report to text file
        with open(f"{save_dir}/classification_report.txt", 'w') as f:
            f.write(report)
        logging.info(f"Classification report saved to {save_dir}/classification_report.txt")
        
        # Save detailed classification metrics to CSV
        report_dict = classification_report(all_labels, all_preds, target_names=target_names, 
                                           zero_division=0, output_dict=True)
        
        # Convert to DataFrame for easier handling
        report_df = pd.DataFrame(report_dict).transpose()
        report_df.to_csv(f"{save_dir}/classification_metrics.csv")
        logging.info(f"Detailed classification metrics saved to {save_dir}/classification_metrics.csv")
    except Exception as e:
        logging.error(f"Error generating classification report: {e}")

    # Compute confusion matrix
    try:
        cm = confusion_matrix(all_labels, all_preds)
        logging.info("\nConfusion Matrix Shape:")
        logging.info(f"{cm.shape}")
        
        # Save raw confusion matrix
        np.save(f"{save_dir}/confusion_matrix.npy", cm)

        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        
        # If there are many classes, use a smaller font and abbreviated labels
        if len(class_names) > 20:
            plt.figure(figsize=(16, 14))
            sns.heatmap(cm, annot=False, cmap="YlGnBu", 
                        xticklabels=[f"C{i}" for i in range(len(class_names))], 
                        yticklabels=[f"C{i}" for i in range(len(class_names))])
            plt.xlabel("Predicted (see class mapping in logs)")
            plt.ylabel("True (see class mapping in logs)")
            
            # Print class mapping
            logging.info("\nClass mapping for confusion matrix:")
            with open(f"{save_dir}/class_mapping.txt", 'w') as f:
                for i, name in enumerate(class_names):
                    mapping = f"C{i} = {name}"
                    logging.info(mapping)
                    f.write(mapping + "\n")
        else:
            sns.heatmap(cm, annot=True, fmt='d', cmap="YlGnBu", 
                        xticklabels=target_names, yticklabels=target_names)
            plt.xlabel("Predicted")
            plt.ylabel("True")
        
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/confusion_matrix.png")
        logging.info(f"Confusion matrix saved to {save_dir}/confusion_matrix.png")
        
        try:
            plt.show()
        except Exception as e:
            logging.warning(f"Could not display plot: {e}")
    except Exception as e:
        logging.error(f"Error generating confusion matrix: {e}")
    
    return accuracy

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Validate a trained model on test data')
    parser.add_argument('--data', type=str, default='data/test', help='Path to test data')
    parser.add_argument('--model', type=str, default='models/best_model.pth', help='Path to trained model')
    parser.add_argument('--output', type=str, default='outputs', help='Directory to save results')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for validation')
    parser.add_argument('--save-predictions', action='store_true', help='Save individual predictions')
    parser.add_argument('--confidence-threshold', type=float, default=0.5, 
                        help='Threshold for reporting low confidence predictions')
    
    args = parser.parse_args()
    
    # Make sure the appropriate imports are available
    try:
        from scripts.dataset import PlantDataset, get_transforms
        from models.CNN import CNN
    except ImportError:
        logging.error("Couldn't import required modules. Make sure scripts/dataset.py and models/CNN.py are available.")
        return

    # Paths
    test_path = args.data
    model_path = args.model
    save_dir = args.output

    # Check if test directory exists
    if not os.path.exists(test_path):
        logging.error(f"Test directory not found: {test_path}")
        return

    logging.info(f"Validating model on test data from: {test_path}")
    logging.info(f"Will save results to: {save_dir}")

    # Get transforms
    try:
        _, val_transform = get_transforms()  # Ensure validation transform matches
    except Exception as e:
        logging.error(f"Error getting transforms: {e}")
        return

    # Load test dataset
    try:
        test_dataset = PlantDataset(test_path, transform=val_transform)
        logging.info(f"Number of test classes: {len(test_dataset.classes)}")
        logging.info(f"Test classes: {test_dataset.classes}")
        logging.info(f"Number of test samples: {len(test_dataset)}")
    except Exception as e:
        logging.error(f"Error loading test dataset: {e}")
        return
    
    # Define test loader
    try:
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2 if torch.cuda.is_available() else 0,
            pin_memory=torch.cuda.is_available()
        )
    except Exception as e:
        logging.error(f"Error creating test loader: {e}")
        return
    
    # Load class names
    model_dir = os.path.dirname(model_path)
    class_names_path = os.path.join(model_dir, "class_names.pt")
    
    if os.path.exists(class_names_path):
        try:
            class_names = torch.load(class_names_path)
            logging.info(f"Class names loaded from '{class_names_path}': {class_names}")
        except Exception as e:
            logging.error(f"Error loading class names: {e}")
            class_names = test_dataset.classes
            logging.info(f"Using class names from dataset: {class_names}")
    else:
        logging.warning(f"'{class_names_path}' not found. Loading class names from the dataset.")
        class_names = test_dataset.classes
        logging.info(f"Class names loaded from dataset: {class_names}")

    # Check if model file exists
    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            logging.info("Model loaded successfully!")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            exit()
    else:
        logging.error("Model file is missing or empty. Ensure training was successful before running validation.")
        exit()

    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")

    # Load the trained model
    try:
        model = CNN(num_classes=len(class_names))
        model.load_state_dict(torch.load(model_path, map_location=device))
        logging.info("\nModel loaded successfully!")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        # Try to load model with different number of classes
        logging.info("Attempting to load model with different class configuration...")
        
        # Check the model's structure to determine the number of classes in the saved model
        try:
            state_dict = torch.load(model_path, map_location=device)
            output_layer_weight = state_dict['classifier.4.weight']  # Updated index to match the new model structure
            saved_num_classes = output_layer_weight.size(0)
            logging.info(f"Saved model has {saved_num_classes} output classes, but dataset has {len(class_names)} classes")
            
            if saved_num_classes != len(class_names):
                logging.warning("Class mismatch between model and dataset!")
                
                # Create a model with the same number of classes as the saved model (for evaluation)
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
    validate_model(
        test_loader, 
        model, 
        class_names, 
        device=device,
        save_dir=save_dir,
        save_predictions=args.save_predictions,
        confidence_threshold=args.confidence_threshold
    )

if __name__ == "__main__":
    main()