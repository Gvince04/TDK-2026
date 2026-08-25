import torch
import torch.nn as nn

class GatedFusionModel(nn.Module):
    def __init__(self, num_dynamic_features=8, num_static_levels=4):
        super(GatedFusionModel, self).__init__()
        
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
        
        self.static_embedding = nn.Embedding(num_embeddings=num_static_levels, embedding_dim=16)
        
        self.gate_generator = nn.Sequential(
            nn.Linear(16, 64),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64 + 16, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1)
        )

    def forward(self, dynamic_x, static_x):
        dyn_features = self.dynamic_encoder(dynamic_x)
        stat_features = self.static_embedding(static_x)
        
        gate = self.gate_generator(stat_features)
        
        gated_dyn = (dyn_features * gate) + dyn_features
        
        fused_features = torch.cat((gated_dyn, stat_features), dim=1)
        
        prediction = self.classifier(fused_features)
        
        return prediction.squeeze(-1)
