import torch
import torch.nn as nn

class GatedFusionModel(nn.Module):
    def __init__(self, num_dynamic_features=4, num_static_levels=4):
        super(GatedFusionModel, self).__init__()

        self.dynamic_encoder = nn.Sequential(
            nn.Conv1d(in_channels=num_dynamic_features, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool1d(1) 
        )
        
        self.static_embedding = nn.Embedding(num_embeddings=num_static_levels, embedding_dim=16)
        
        self.gate_generator = nn.Sequential(
            nn.Linear(16, 64),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64 + 16, 32),
            nn.ReLU(),
            nn.Dropout(0.3), 
            nn.Linear(32, 1) 
        )

    def forward(self, dynamic_x, static_x):
        dyn_features = self.dynamic_encoder(dynamic_x)
        dyn_features = dyn_features.squeeze(-1)
        
        stat_features = self.static_embedding(static_x)
        
        gate_weights = self.gate_generator(stat_features)
        
        gated_dyn_features = dyn_features * gate_weights

        combined_features = torch.cat((gated_dyn_features, stat_features), dim=1)
        
        prediction = self.classifier(combined_features)
        
        return prediction.squeeze(-1)