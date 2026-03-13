import torch.nn.modules as nn

from infra.data_models import ConfigCNN


class MFCC_CNN(nn.Module):
    def __init__(self, *, cfg: ConfigCNN):
        super().__init__()

        ...
