import torch
import torch.nn as nn

class BaselineModel(nn.Module):
    def __init__(self, num_dynamic_features=10):
        super(BaselineModel, self).__init__()
        
        self.dynamic_encoder = nn.Sequential(
            nn.Linear(num_dynamic_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1)
        )

    def forward(self, dynamic_x, static_x=None):
        dyn_features = self.dynamic_encoder(dynamic_x)
        prediction = self.classifier(dyn_features)
        return prediction.squeeze(-1)
