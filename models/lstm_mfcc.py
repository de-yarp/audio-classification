import torch
import torch.nn as nn

from infra.data_models import ConfigLSTM


class MFCC_LSTM(nn.Module):
    def __init__(self, *, cfg: ConfigLSTM):
        super().__init__()
        self.input_size = cfg.input_size
        self.lstm = nn.LSTM(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.classifier = nn.ModuleList()
        fc_input_size = cfg.hidden_size
        for fc_output_size in cfg.fc_layers:
            self.classifier.append(nn.Linear(fc_input_size, fc_output_size))
            fc_input_size = fc_output_size
        self.classifier.append(nn.Linear(fc_input_size, cfg.num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        _, (hidden_n, _) = self.lstm(x)
        last_hidden = hidden_n[-1]

        x = last_hidden
        for fc in self.classifier[:-1]:
            x = torch.relu(fc(x))

        return self.classifier[-1](x)
