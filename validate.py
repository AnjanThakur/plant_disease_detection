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

# ✅ Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("validation.log"), logging.StreamHandler()],
)

def validate_model(
    test_loader,
    model,
    class_names,
    device="cpu",
    save_dir="outputs",
    save_predictions=False,
    confidence_threshold=0.5,
):
    """
    ✅ Validate the trained model on the test dataset and compute metrics.
    """
    logging.info(f"Validating on {device}")

    # ✅ Create output directory
    os.makedirs(save_dir, exist_ok=True)

    model.to(device)
    model.eval()  # ✅ Set model to evaluation mode

    all_preds = []
    all_labels = []
    all_confidences = []
    all_paths = []

    # ✅ Class-wise predictions
    class_correct = {i: 0 for i in range(len(class_names))}
    class_total = {i: 0 for i in range(len(class_names))}

    # ✅ Process batches with a progress bar
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Validating"):
            # ✅ Handle batch format correctly
            if isinstance(batch, tuple) and len(batch) >= 2:
                images = batch[0].to(device)
                labels = batch[1].to(device)

                # ✅ Check for paths if available
                paths = batch[2] if len(batch) >= 3 and isinstance(batch[2], list) else None
            else:
                logging.error(f"Unexpected batch format: {batch}")
                continue

            # ✅ Get model predictions
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)

            # ✅ Get predicted class and confidence
            confidence_values, predicted = torch.max(probabilities, 1)

            # ✅ Extend prediction lists
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidence_values.cpu().numpy())

            if paths is not None:
                all_paths.extend(paths)

            # ✅ Class-wise accuracy tracking
            for i in range(len(labels)):
                label = labels[i].item()
                pred = predicted[i].item()

                if label in class_total:
                    class_total[label] += 1
                    if label == pred:
                        class_correct[label] += 1
                else:
                    logging.warning(f"Found label {label} which is out of range for class_names")

    # ✅ Handle empty predictions gracefully
    if len(all_preds) == 0 or len(all_labels) == 0:
        logging.error("No valid predictions or labels found! Exiting validation...")
        return 0.0

    # ✅ Compute overall accuracy
    accuracy = sum(class_correct.values()) / sum(class_total.values())
    logging.info(f"Overall Accuracy: {accuracy * 100:.2f}%")

    # ✅ Class-wise accuracy and saving
    class_accuracy_data = []
    for i in range(len(class_names)):
        if class_total.get(i, 0) > 0:
            acc = class_correct.get(i, 0) / class_total.get(i, 0)
            logging.info(f"{class_names[i]}: {acc * 100:.2f}% ({class_correct.get(i, 0)}/{class_total.get(i, 0)})")

            class_accuracy_data.append(
                {
                    "class": class_names[i],
                    "accuracy": acc * 100,
                    "correct": class_correct.get(i, 0),
                    "total": class_total.get(i, 0),
                }
            )
        else:
            logging.warning(f"{class_names[i]}: No test samples")

    # ✅ Save class-wise accuracy to CSV
    accuracy_df = pd.DataFrame(class_accuracy_data)
    accuracy_df.to_csv(f"{save_dir}/class_accuracy.csv", index=False)
    logging.info(f"Class-wise accuracy saved to {save_dir}/class_accuracy.csv")

    # ✅ Plot class-wise accuracy
    plt.figure(figsize=(15, 10))
    ordered_df = accuracy_df.sort_values("accuracy", ascending=False)
    plt.xticks(rotation=90, fontsize=10)
    sns.barplot(x="class", y="accuracy", data=ordered_df)
    plt.title("Class-wise Accuracy")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/class_accuracy.png")

    # ✅ Classification Report
    try:
        report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
        with open(f"{save_dir}/classification_report.txt", "w") as f:
            f.write(report)
        logging.info(f"Classification report saved to {save_dir}/classification_report.txt")
    except Exception as e:
        logging.error(f"Error generating classification report: {e}")

    # ✅ Confusion Matrix
    try:
        cm = confusion_matrix(all_labels, all_preds)
        np.save(f"{save_dir}/confusion_matrix.npy", cm)

        plt.figure(figsize=(12, 10))
        if len(class_names) > 1:
            sns.heatmap(cm, annot=len(class_names) <= 20, cmap="YlGnBu",
                        xticklabels=class_names, yticklabels=class_names)
        else:
            sns.heatmap(cm, annot=True, cmap="YlGnBu")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/confusion_matrix.png")
        logging.info(f"Confusion matrix saved to {save_dir}/confusion_matrix.png")
    except Exception as e:
        logging.error(f"Error generating confusion matrix: {e}")

    return accuracy


# ✅ Main Function
def main():
    # ✅ Argument Parser
    parser = argparse.ArgumentParser(description="Validate a trained model on test data")
    parser.add_argument("--data", type=str, default="data/test", help="Path to test data")
    parser.add_argument("--model", type=str, default="best_model.pth", help="Path to trained model")
    parser.add_argument("--output", type=str, default="outputs", help="Directory to save results")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for validation")

    args = parser.parse_args()

    try:
        # ✅ Import required modules dynamically
        from scripts.dataset import PlantDataset, get_transforms
        from models.CNN import CNN
    except ImportError:
        logging.error("Couldn't import required modules. Check that scripts/dataset.py and models/CNN.py exist.")
        return

    # ✅ Paths
    test_path = args.data
    model_path = args.model
    save_dir = args.output

    # ✅ Check if test directory exists
    if not os.path.exists(test_path):
        logging.error(f"Test directory not found: {test_path}")
        return

    # ✅ Get test transforms
    try:
        _, val_transform = get_transforms()
    except Exception as e:
        logging.error(f"Error getting transforms: {e}")
        return

    # ✅ Load test dataset
    try:
        test_dataset = PlantDataset(test_path, transform=val_transform)
    except Exception as e:
        logging.error(f"Error loading test dataset: {e}")
        return

    # ✅ Define test loader
    try:
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
    except Exception as e:
        logging.error(f"Error creating test loader: {e}")
        return

    # ✅ Load class names if available
    model_dir = os.path.dirname(model_path)
    class_names_path = os.path.join(model_dir, "class_names.pt")

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

    # ✅ Load Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = CNN(num_classes=len(class_names))  # ✅ Ensure correct number of output classes
        if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
            model.load_state_dict(torch.load(model_path, map_location=device))
            logging.info("Model loaded successfully!")
        else:
            logging.error("Model file missing or empty. Ensure training was successful before running validation.")
            exit()
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        exit()

    # ✅ Validate model
    validate_model(
        test_loader,
        model,
        class_names,
        device=device,
        save_dir=save_dir,
    )


if __name__ == "__main__":
    main()
