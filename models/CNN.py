import torch
import torch.nn as nn
import torchvision.models as models

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()

        # Using pre-trained ResNet50 as feature extractor
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Remove the fully connected layers (classifier part) from ResNet
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # Add new classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(resnet.fc.in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
