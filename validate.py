import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from scripts.dataset import PlantDataset, get_transforms
from models.CNN import CNN

def validate_model(test_loader, model, class_names):
    """
    Validate the trained model on the test dataset and compute metrics.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap="YlGnBu")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

def main():
    # Paths
    test_path = 'data/test'

    print("Validating on CPU")

    # Get transforms
    _, val_transform = get_transforms()  # Ensure validation transform matches

    # Load test dataset
    test_dataset = PlantDataset(test_path, transform=val_transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,  # Match batch size used during training
        shuffle=False,
        num_workers=2
    )

    # Get class names from the dataset
    class_names = test_dataset.classes
    print(f"Class names from dataset: {class_names}")
    print(f"Number of class names: {len(class_names)}")

    # Load the trained model
    model = CNN(num_classes=len(class_names))  # Make sure this matches the dataset
    model.load_state_dict(torch.load('C:/Users/HP/Desktop/plant_disease_detection/best_model.pth', map_location=torch.device('cpu')))
    print("\nModel loaded successfully!")

    # Validate the model
    validate_model(test_loader, model, class_names)

if __name__ == "__main__":
    main()
