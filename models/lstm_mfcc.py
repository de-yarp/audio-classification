import torch
import torch.nn as nn

from infra.data_models import ConfigLSTM


class MFCC_LSTM(nn.Module):
    def __init__(self, *, cfg: ConfigLSTM):
        super().__init__()
        self.input_size = 120 if cfg.mfcc_deltas else 40
        self.pooling = cfg.pooling
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            batch_first=True,
            bidirectional=cfg.bidirectional,
        )
        self.classifier = nn.ModuleList()
        fc_input_size = cfg.hidden_size * (2 if cfg.bidirectional else 1)
        for fc_output_size in cfg.fc_layers:
            self.classifier.append(nn.Linear(fc_input_size, fc_output_size))
            fc_input_size = fc_output_size
        self.classifier.append(nn.Linear(fc_input_size, cfg.num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # stacked-as-channels case
        if x.ndim == 4:
            # (batch, 3, 40, T) → (batch, 120, T)
            batch, channels, coeffs, time = x.shape
            x = x.reshape(batch, channels * coeffs, time)

        # Expected shape is (batch_size, input_size, time_steps).
        if x.ndim != 3 or x.shape[1] != self.input_size:
            msg = (
                "expected input shape convention "
                f"(batch_size, input_size, time_steps) with input_size={self.input_size}, "
                f"got {tuple(x.shape)}"
            )
            raise ValueError(msg)

        x = x.transpose(1, 2)

        # LSTM expects (batch_size, seq_len, input_size), so this becomes
        # (batch_size, 216, 120).
        output, (hidden_n, _) = self.lstm(x)

        if self.pooling == "last":
            # h_n shape: (num_layers * num_directions, batch, hidden_size)
            # bidirectional last layer: forward=h_n[-2], backward=h_n[-1]
            if self.lstm.bidirectional:
                x = torch.cat([hidden_n[-2], hidden_n[-1]], dim=1)
            else:
                x = hidden_n[-1]
        elif self.pooling == "mean":
            x = output.mean(dim=1)
        elif self.pooling == "max":
            x = output.max(dim=1).values

        for fc in self.classifier[:-1]:
            x = torch.relu(fc(x))

        return self.classifier[-1](x)
