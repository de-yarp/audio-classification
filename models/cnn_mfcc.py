import torch
import torch.nn.functional as F
import torch.nn.modules as nn

from infra.data_models import ConfigCNN


class MFCC_CNN(nn.Module):
    def __init__(self, *, cfg: ConfigCNN):
        super().__init__()

        self.conv1 = nn.Conv2d(  # (1, 120, 216)
            1,
            cfg.conv_kernel_count,
            cfg.conv_kernel_size,
            cfg.conv_stride,
            cfg.conv_padding,
        )
        self.pool = nn.MaxPool2d(  # (16, 118, 214)
            cfg.pool_kernel_size, cfg.pool_stride, cfg.pool_padding
        )
        self.conv2 = nn.Conv2d(  # (16, 59, 107)
            cfg.conv_kernel_count,
            2 * cfg.conv_kernel_count,
            cfg.conv_kernel_size,
            cfg.conv_stride,
            cfg.conv_padding,
        )  # (32, 57, 105)
        # (32, 28, 52)
        self.fc1 = nn.Linear(32 * 28 * 52, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 50)

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x
