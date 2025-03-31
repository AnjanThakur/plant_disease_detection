import torch
import torch.nn as nn
import torchvision.models as models

class CNN(nn.Module):
    def __init__(self, num_classes=38):
        super(CNN, self).__init__()
        self.features = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # Freeze feature extractor
        for param in self.features.parameters():
            param.requires_grad = False

        # Modify the fully connected layer (fc) of ResNet50
        in_features = self.features.fc.in_features
        self.features.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return x

    def freeze_feature_extractor(self):
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_feature_extractor(self):
        for param in self.features.parameters():
            param.requires_grad = True
