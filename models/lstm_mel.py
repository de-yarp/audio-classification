import torch.nn as nn

from infra.data_models import ConfigLSTM


class MEL_LSTM(nn.Module):
    def __init__(self, *, cfg: ConfigLSTM):
        super().__init__()
        ...
