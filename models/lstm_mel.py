import torch
import torch.nn as nn
from infra.data_models import ConfigLSTM

class MEL_LSTM(nn.Module):
    def __init__(self, *, cfg: ConfigLSTM):
        super().__init__()
        
        hidden_size = cfg.hidden_size
        num_layers = cfg.num_layers
        
        input_size = getattr(cfg, 'n_mels', 128) 
        
        num_classes = 50 
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=getattr(cfg, 'dropout', 0.3) if num_layers > 1 else 0.0
        )
        
        
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):

        if x.shape[1] == 128 and x.shape[2] != 128:
            x = x.transpose(1, 2)
            
        lstm_out, _ = self.lstm(x)
        
        pooled_out = lstm_out.mean(dim=1)
        
        out = self.fc(pooled_out)
        
        return out