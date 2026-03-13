import torch.nn as nn

from infra.data_models import ConfigCNN


class MEL_CNN(nn.Module):
    def __init__(self, *, cfg: ConfigCNN):
        super().__init__()
        ...
