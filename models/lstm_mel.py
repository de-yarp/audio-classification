import torch
import torch.nn as nn
from infra.data_models import ConfigLSTM

class MEL_LSTM(nn.Module):
    """Mel-spectrogram LSTM classifier."""
    
    def __init__(self, *, cfg: ConfigLSTM):
        super().__init__()
        
        # 1. Config & parameters
        hidden_size = cfg.hidden_size
        num_layers = cfg.num_layers
        num_classes = cfg.num_classes
        
        self.input_size = getattr(cfg, 'n_mels', 128) 
        self.is_bidirectional = getattr(cfg, 'bidirectional', False)
        self.pooling = getattr(cfg, 'pooling', 'last') 
        
        # 2. Input normalization
        self.input_bn = nn.BatchNorm1d(self.input_size)
        
        # 3. LSTM block
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=getattr(cfg, 'dropout', 0.0) if num_layers > 1 else 0.0,
            bidirectional=self.is_bidirectional
        )
        
        # 4. Classifier fully-connected (FC) layers
        lstm_out_size = hidden_size * 2 if self.is_bidirectional else hidden_size
        if self.pooling == 'mean_max':
            lstm_out_size *= 2 
            
        fc_layers_list = getattr(cfg, 'fc_layers', [])
        dropout_rate = getattr(cfg, 'dropout', 0.0)
        
        layers = [nn.LayerNorm(lstm_out_size)]
        in_features = lstm_out_size
        
        for out_features in fc_layers_list:
            layers.extend([
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),
                nn.ReLU()
            ])
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_features = out_features
            
        layers.append(nn.Linear(in_features, num_classes))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        # 1. Shape correction & norm: requires (Batch, Channels, Time)
        if x.shape[1] == self.input_size and x.shape[2] != self.input_size:
            x = x.transpose(1, 2) 
            
        x = x.transpose(1, 2)
        x = self.input_bn(x)
        x = x.transpose(1, 2)
            
        # 2. LSTM pass
        lstm_out, _ = self.lstm(x)
        
        # 3. Pooling over time dimension
        if self.pooling == 'mean_max':
            out_features = torch.cat((lstm_out.mean(dim=1), lstm_out.max(dim=1)[0]), dim=1)
        elif self.pooling == 'mean':
            out_features = lstm_out.mean(dim=1)
        elif self.pooling == 'max':
            out_features = lstm_out.max(dim=1)[0]
        else:
            out_features = lstm_out[:, -1, :] 
        
        # 4. Final classification
        return self.fc(out_features)