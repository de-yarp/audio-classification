from math import floor

import torch
import torch.nn.functional as F
import torch.nn.modules as nn

from infra.data_models import ConfigCNN, LayerConv, LayerPool, PoolType


class MFCC_CNN(nn.Module):
    def __init__(self, *, cfg: ConfigCNN):
        super().__init__()

        self.activation_fn = F.relu
        self.num_classes = cfg.num_classes
        self.conv_layers_list = nn.ModuleList()
        self.fc_layers_list = nn.ModuleList()
        self.dropout = cfg.dropout
        out_channels = 1
        height = 120 if cfg.mfcc_deltas else 40
        width = 216
        pool2d = None
        if cfg.pool_type == PoolType.MAX:
            pool2d = nn.MaxPool2d
        elif cfg.pool_type == PoolType.AVG:
            pool2d = nn.AvgPool2d
        else:
            msg = f"invalid model.pool_type {cfg.pool_type}, expected {[e.value for e in PoolType]}"
            raise ValueError(msg)

        for layer in cfg.conv_layers:
            kernel_count = -1
            if isinstance(layer, LayerConv):
                self.conv_layers_list.append(
                    nn.Conv2d(
                        out_channels,
                        layer.kernel_count,
                        layer.kernel_size,
                        layer.stride,
                        layer.padding,
                    )
                )
                if layer.batch_norm:
                    self.conv_layers_list.append(nn.BatchNorm2d(layer.kernel_count))

                kernel_count = layer.kernel_count
            if isinstance(layer, LayerPool):
                self.conv_layers_list.append(
                    pool2d(
                        layer.kernel_size,
                        layer.stride,
                        layer.padding,
                    )
                )
                kernel_count = out_channels

            out_channels, height, width = self._post_transform_shape(
                kernel_count,
                layer.kernel_size,
                layer.stride,
                layer.padding,
                width,
                height,
            )

        flat_tensor_shape = out_channels * height * width
        fc_layers = cfg.fc_layers

        last_fc_out_channels = fc_layers[0]
        self.fc_layers_list.append(nn.Linear(flat_tensor_shape, last_fc_out_channels))
        if self.dropout != 0.0:
            self.fc_layers_list.append(nn.Dropout(self.dropout))
        for fc_out_channels in fc_layers[1:]:
            self.fc_layers_list.append(nn.Linear(last_fc_out_channels, fc_out_channels))
            if self.dropout != 0.0:
                self.fc_layers_list.append(nn.Dropout(self.dropout))
            last_fc_out_channels = fc_out_channels
        self.fc_layers_list.append(nn.Linear(last_fc_out_channels, self.num_classes))

    def _post_transform_shape(
        self,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        width: int,
        height: int,
    ):
        new_height = floor((height - kernel_size + 2 * padding) / stride) + 1
        new_width = floor((width - kernel_size + 2 * padding) / stride) + 1
        return (out_channels, new_height, new_width)

    def forward(self, x: torch.Tensor):
        def _is_next_batch_norm(conv_layer_list: nn.ModuleList, idx: int) -> bool:
            try:
                return isinstance(conv_layer_list[idx + 1], nn.BatchNorm2d)
            except IndexError:
                return False

        x = x.unsqueeze(1)
        for idx, conv in enumerate(self.conv_layers_list):
            # conv with batch norm next → no relu. conv without batch norm next → relu. batch norm → relu (falls through to last line). pool → no relu
            if isinstance(conv, nn.MaxPool2d) or _is_next_batch_norm(
                self.conv_layers_list, idx
            ):
                x = conv(x)
                continue
            x = self.activation_fn(conv(x))

        x = torch.flatten(x, 1)

        for fc in self.fc_layers_list[:-1]:
            if isinstance(fc, nn.Dropout):
                x = fc(x)
                continue
            x = self.activation_fn(fc(x))

        x = self.fc_layers_list[-1](x)

        return x
