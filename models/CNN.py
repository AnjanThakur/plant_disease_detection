import torch
import torch.nn as nn
import torchvision.models as models

class CNN(nn.Module):
    def __init__(self, num_classes=38):
        super(CNN, self).__init__()
        self.features = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        in_features = self.features.fc.in_features
        self.features.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.features(x)

    def load_weights(self, model_path, device="cpu"):
        state_dict = torch.load(model_path, map_location=device)
        model_keys = state_dict.keys()

        if "model_state_dict" in model_keys:
            state_dict = state_dict["model_state_dict"]

        fc_weights = [k for k in state_dict.keys() if "fc" in k]
        if len(fc_weights) > 0:
            expected_fc_out = self.features.fc[-1].out_features
            loaded_fc_out = state_dict[fc_weights[-1]].shape[0]

            if expected_fc_out != loaded_fc_out:
                print(f"⚠️ Adjusting FC layer from {loaded_fc_out} to {expected_fc_out}")
                in_features = self.features.fc[0].in_features
                self.features.fc = nn.Sequential(
                    nn.Linear(in_features, 512),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(512, expected_fc_out)
                )

        self.load_state_dict(state_dict)

    def freeze_feature_extractor(self):
        for param in self.features.parameters():
            param.requires_grad = False
    
    def unfreeze_feature_extractor(self):
        for param in self.features.parameters():
            param.requires_grad = True