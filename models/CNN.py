import torch
import torch.nn as nn
from torchvision import models

class CNN(nn.Module):
    def __init__(self, num_classes, freeze_layers=False, pretrained=True):
        """
        CNN class with a customizable fully connected layer based on ResNet50.
        
        Args:
        - num_classes: The number of output classes.
        - freeze_layers: Whether to freeze the convolutional layers (default: False).
        - pretrained: Whether to use pretrained weights for the model (default: True).
        """
        super(CNN, self).__init__()

        # Load ResNet50 with or without pre-trained weights
        self.features = models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)

        # Adjust the final fully connected layer to match the number of output classes
        in_features = self.features.fc.in_features
        self.features.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout for regularization
            nn.Linear(512, num_classes)  # Output layer based on the number of classes
        )

        # Freeze or unfreeze layers based on the flag `freeze_layers`
        if freeze_layers:
            self.freeze_layers()

    def forward(self, x):
        """
        Forward pass through the model
        """
        return self.features(x)

    def freeze_layers(self):
        """
        Freeze the convolutional layers so they don't update during training.
        Only the fully connected layers (FC) are trainable.
        """
        for param in self.features.parameters():
            param.requires_grad = False

        # Ensure the final fully connected layer's parameters are trainable
        for param in self.features.fc.parameters():
            param.requires_grad = True

    def unfreeze_layers(self):
        """
        Unfreeze all layers so that all the layers will be updated during training.
        """
        for param in self.features.parameters():
            param.requires_grad = True

    def load_weights(self, model_path, device="cpu"):
        """
        Load the weights into the model.
        This method ensures the model is loaded properly and checks if the FC layer needs adjustments.
        
        Args:
        - model_path: The path to the saved model weights.
        - device: The device to load the model onto ("cpu" or "cuda").
        """
        state_dict = torch.load(model_path, map_location=device)
        
        # Ensure model_state_dict is available in some formats
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        # Remove FC layer weights to prevent mismatch
        fc_weight_keys = [k for k in state_dict.keys() if "fc" in k]
        for key in fc_weight_keys:
            del state_dict[key]

        # Load state dict for the rest of the model, excluding the FC layer
        self.features.load_state_dict(state_dict, strict=False)

        # Reinitialize the FC layer with the correct number of output classes
        in_features = self.features.fc[0].in_features
        self.features.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)  # Ensure the number of classes matches
        )

    def get_fc_parameters(self):
        """
        Returns the parameters of the final fully connected layers.
        """
        return self.features.fc.parameters()
