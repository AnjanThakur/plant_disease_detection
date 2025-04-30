import torch
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import logging
import os
from tqdm import tqdm
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch.nn.functional as F

def setup_logging():
    """Configure logging settings."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("validation.log"), logging.StreamHandler()],
    )

def validate_model(test_loader, model, class_names, device="cpu", save_dir="outputs"):
    """Validate the trained model and compute metrics."""
    logging.info(f"Validating on {device}")
    os.makedirs(save_dir, exist_ok=True)
    model.to(device)
    model.eval()

    all_preds, all_labels, all_confidences, all_paths = [], [], [], []
    class_correct, class_total = {i: 0 for i in range(len(class_names))}, {i: 0 for i in range(len(class_names))}

    total_loss = 0.0
    all_losses = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Validating"):
            try:
                if len(batch) == 3:
                    images, labels, paths = batch
                elif len(batch) == 2:
                    images, labels = batch
                    paths = ["unknown"] * len(labels)
                else:
                    logging.error(f"Unexpected batch format with {len(batch)} elements: {batch}")
                    continue

                images, labels = images.to(device), labels.to(device)
            except Exception as e:
                logging.error(f"Error unpacking batch: {e}")
                continue

            outputs = model(images)
            if outputs is None or outputs.nelement() == 0:
                logging.error("Empty output from model! Skipping batch.")
                continue

            probabilities = F.softmax(outputs, dim=1)
            if torch.isnan(probabilities).any():
                logging.error("NaN values found in model output! Skipping batch.")
                continue

            # Calculate loss
            loss = F.cross_entropy(outputs, labels)
            total_loss += loss.item()
            all_losses.append(loss.item())

            confidence_values, predicted = torch.max(probabilities, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidence_values.cpu().numpy())
            all_paths.extend(paths)

            for i in range(len(labels)):
                label, pred = labels[i].item(), predicted[i].item()
                if label in class_total:
                    class_total[label] += 1
                    if label == pred:
                        class_correct[label] += 1
                else:
                    logging.warning(f"Unexpected label {label} found.")

    if not all_preds or not all_labels:
        logging.error("No valid predictions or labels found! Exiting validation...")
        return 0.0

    accuracy = sum(class_correct.values()) / max(sum(class_total.values()), 1)
    logging.info(f"Overall Accuracy: {accuracy * 100:.2f}%")

    # Average loss
    avg_loss = total_loss / len(all_losses)
    logging.info(f"Average Loss: {avg_loss:.4f}")

    accuracy_df = pd.DataFrame([
        {"class": class_names[i], "accuracy": class_correct.get(i, 0) / max(class_total.get(i, 1), 1) * 100,
         "correct": class_correct.get(i, 0), "total": class_total.get(i, 0)}
        for i in range(len(class_names))
    ])
    accuracy_df.to_csv(f"{save_dir}/class_accuracy.csv", index=False)
    logging.info(f"Class-wise accuracy saved to {save_dir}/class_accuracy.csv")

    try:
        report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
        with open(f"{save_dir}/classification_report.txt", "w") as f:
            f.write(report)
        logging.info(f"Classification report saved to {save_dir}/classification_report.txt")
    except Exception as e:
        logging.error(f"Error generating classification report: {e}")

    # Calculate precision, recall, f1-score
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    logging.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    # === Plot and save accuracy bar graph ===
    try:
        plt.figure(figsize=(12, 6))
        sns.barplot(x="class", y="accuracy", data=accuracy_df)
        plt.xticks(rotation=90)
        plt.title("Class-wise Accuracy")
        plt.tight_layout()
        plot_path = os.path.join(save_dir, "class_accuracy_plot.png")
        plt.savefig(plot_path)
        logging.info(f"Accuracy plot saved to {plot_path}")
        plt.close()
    except Exception as e:
        logging.error(f"Failed to plot class-wise accuracy: {e}")

    # === Plot and save confusion matrix ===
    try:
        cm = confusion_matrix(all_labels, all_preds)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        plt.title("Normalized Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.xticks(rotation=90)
        plt.tight_layout()
        cm_path = os.path.join(save_dir, "confusion_matrix.png")
        plt.savefig(cm_path)
        logging.info(f"Confusion matrix saved to {cm_path}")
        plt.close()
    except Exception as e:
        logging.error(f"Failed to generate confusion matrix: {e}")

    return accuracy

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Validate a trained model on test data")
    parser.add_argument("--data", type=str, default="data/Tomato/test_tomato", help="Path to test data")
    parser.add_argument("--model", type=str, default="models/best_tomato_model.pth", help="Path to trained model")
    parser.add_argument("--output", type=str, default="outputs", help="Directory to save results")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for validation")

    args = parser.parse_args()

    try:
        from scripts.dataset import PlantDataset, get_transforms
        from models.CNN import CNN
    except ImportError as e:
        logging.error(f"Couldn't import required modules. Check scripts/dataset.py and models/CNN.py. Error: {e}")
        return

    test_path, model_path, save_dir = args.data, args.model, args.output
    if not os.path.exists(test_path):
        logging.error(f"Test directory not found: {test_path}")
        return

    _, val_transform = get_transforms()
    test_dataset = PlantDataset(test_path, transform=val_transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    class_names_path = os.path.join(os.path.dirname(model_path), "class_names.pt")
    if os.path.exists(class_names_path):
        try:
            class_names = torch.load(class_names_path)
            logging.info(f"Class names loaded from '{class_names_path}': {class_names}")
        except Exception as e:
            logging.error(f"Error loading class names: {e}")
            class_names = test_dataset.classes
    else:
        logging.warning(f"'{class_names_path}' not found. Using dataset class names.")
        class_names = test_dataset.classes

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CNN(num_classes=len(class_names))

    if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
        logging.error("Model file missing or empty. Ensure training was successful before running validation.")
        raise SystemExit

    model.load_state_dict(torch.load(model_path, map_location=device))
    logging.info("Model loaded successfully!")

    validate_model(test_loader, model, class_names, device=device, save_dir=save_dir)

if __name__ == "__main__":
    main()
