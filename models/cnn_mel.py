from math import floor

import torch
import torch.nn.functional as F
import torch.nn.modules as nn

from infra.data_models import ConfigCNN, LayerConv, LayerPool, PoolType

class MEL_CNN(nn.Module):
    def __init__(self, *, cfg: ConfigCNN):
        super().__init__()

        self.activation_fn = F.relu
        self.num_classes = cfg.num_classes
        self.conv_layers_list = nn.ModuleList()
        self.fc_layers_list = nn.ModuleList()
        self.dropout = cfg.dropout
        self.global_avg_pool: nn.AdaptiveAvgPool2d | None = None

        in_channels = 1 
        height = 128  
        width = 216   
        
        pool2d = {
            PoolType.MAX: nn.MaxPool2d,
            PoolType.AVG: nn.AvgPool2d
        }.get(cfg.pool_type)

        if pool2d is None:
            msg = f"invalid model.pool_type {cfg.pool_type}, expected {[e.value for e in PoolType]}"
            raise ValueError(msg)

        current_channels = in_channels
        h, w = height, width

        for layer in cfg.conv_layers:
            if isinstance(layer, LayerConv):
                self.conv_layers_list.append(
                    nn.Conv2d(
                        in_channels=current_channels,
                        out_channels=layer.kernel_count,
                        kernel_size=layer.kernel_size,
                        stride=layer.stride,
                        padding=layer.padding,
                    )
                )
                if layer.batch_norm:
                    self.conv_layers_list.append(nn.BatchNorm2d(layer.kernel_count))
                
                current_channels = layer.kernel_count
                h, w = self._calculate_size(h, w, layer.kernel_size, layer.stride, layer.padding)
                
            elif isinstance(layer, LayerPool):
                self.conv_layers_list.append(
                    pool2d(kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding)
                )
                h, w = self._calculate_size(h, w, layer.kernel_size, layer.stride, layer.padding)

        if cfg.global_avg_pool is not None:
            self.global_avg_pool = nn.AdaptiveAvgPool2d(tuple(cfg.global_avg_pool))
        else:
            self.global_avg_pool = None
        fc_input_size = current_channels

        for fc_size in cfg.fc_layers:
            self.fc_layers_list.append(nn.Linear(fc_input_size, fc_size))
            fc_input_size = fc_size
        
        self.output_layer = nn.Linear(fc_input_size, self.num_classes)

    def _calculate_size(self, h, w, k, s, p):
        kh, kw = (k, k) if isinstance(k, int) else k
        sh, sw = (s, s) if isinstance(s, int) else s
        ph, pw = (p, p) if isinstance(p, int) else p
        
        new_h = floor((h - kh + 2 * ph) / sh) + 1
        new_w = floor((w - kw + 2 * pw) / sw) + 1
        return new_h, new_w

    def forward(self, x: torch.Tensor):
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        for i, layer in enumerate(self.conv_layers_list):
            if isinstance(layer, (nn.BatchNorm2d, nn.MaxPool2d, nn.AvgPool2d)):
                x = layer(x)
            else:
                x = self.activation_fn(layer(x))

        if self.global_avg_pool is not None:
            x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for fc in self.fc_layers_list:
            x = self.activation_fn(fc(x))
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.output_layer(x)