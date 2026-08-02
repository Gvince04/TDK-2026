import torch
import torch.nn as nn

class BaselineModel(nn.Module):
    def __init__(self, num_dynamic_features=4):
        super(BaselineModel, self).__init__()
        
        # 1. DINAMIKUS ÁG (Ugyanaz, mint a fúziós modellnél)
        self.dynamic_encoder = nn.Sequential(
            nn.Conv1d(in_channels=num_dynamic_features, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool1d(1) 
        )
        
        # 2. OSZTÁLYOZÓ FEJ (Kizárólag a 64 dimenziós dinamikus adatot kapja)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

    # A static_x paramétert meghagyjuk a függvényben, hogy ne kelljen 
    # átírni a train.py adagolóját, de a modell egyáltalán nem használja!
    def forward(self, dynamic_x, static_x=None):
        
        # Csak a dinamikus adatot kódoljuk
        dyn_features = self.dynamic_encoder(dynamic_x) 
        dyn_features = dyn_features.squeeze(-1)        
        
        # Egyenesen az osztályozóba megy (nincs fúzió!)
        prediction = self.classifier(dyn_features) 
        
        return prediction.squeeze(-1)