import torch
import torch.nn as nn
import torchvision.models as models

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()

        # Using pre-trained ResNet50 as feature extractor
        try:
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        except AttributeError:
            # Fallback for older PyTorch versions
            resnet = models.resnet50(pretrained=True)

        # Remove the fully connected layers (classifier part) from ResNet
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # Add new classifier layers with batch normalization for better training stability
        self.classifier = nn.Sequential(
            nn.Linear(resnet.fc.in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights properly
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        # Initialize the new classifier layers
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def freeze_feature_extractor(self):
        """Freeze the feature extractor layers to fine-tune only the classifier"""
        for param in self.features.parameters():
            param.requires_grad = False
            
    def unfreeze_feature_extractor(self):
        """Unfreeze the feature extractor layers for full model training"""
        for param in self.features.parameters():
            param.requires_grad = True
            
    def get_output_layer_name(self):
        """Return the name of the output layer weight for model inspection"""
        return 'classifier.4.weight'