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
        self.global_avg_pool: nn.AdaptiveAvgPool2d | None = None
        out_channels = 3 if (cfg.mfcc_deltas and cfg.stack_deltas_as_channels) else 1
        height = 120 if (cfg.mfcc_deltas and not cfg.stack_deltas_as_channels) else 40
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

            kernel_size: int | list[int] = layer.kernel_size
            if isinstance(kernel_size, list):
                kernel_size: tuple[int, int] = tuple(kernel_size)

            kernel_stride: int | list[int] = layer.stride
            if isinstance(kernel_stride, list):
                kernel_stride: tuple[int, int] = tuple(kernel_stride)

            if isinstance(layer, LayerConv):
                self.conv_layers_list.append(
                    nn.Conv2d(
                        out_channels,
                        layer.kernel_count,
                        kernel_size,
                        kernel_stride,
                        layer.padding,
                    )
                )
                if layer.batch_norm:
                    self.conv_layers_list.append(nn.BatchNorm2d(layer.kernel_count))

                kernel_count = layer.kernel_count
            if isinstance(layer, LayerPool):
                self.conv_layers_list.append(
                    pool2d(
                        kernel_size,
                        kernel_stride,
                        layer.padding,
                    )
                )
                kernel_count = out_channels

            out_channels, height, width = self._post_transform_shape(
                kernel_count,
                kernel_size,
                kernel_stride,
                layer.padding,
                width,
                height,
            )

        if cfg.global_avg_pool is not None:
            self.global_avg_pool = nn.AdaptiveAvgPool2d(tuple(cfg.global_avg_pool))
            flat_tensor_shape = (
                out_channels * cfg.global_avg_pool[0] * cfg.global_avg_pool[1]
            )
        else:
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
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int],
        padding: int,
        width: int,
        height: int,
    ):
        kH, kW = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        )
        sH, sW = (stride, stride) if isinstance(stride, int) else stride
        new_height = floor((height - kH + 2 * padding) / sH) + 1
        new_width = floor((width - kW + 2 * padding) / sW) + 1
        return (out_channels, new_height, new_width)

    def forward(self, x: torch.Tensor):
        def _is_next_batch_norm(conv_layer_list: nn.ModuleList, idx: int) -> bool:
            try:
                return isinstance(conv_layer_list[idx + 1], nn.BatchNorm2d)
            except IndexError:
                return False

        if x.dim() == 3:
            x = x.unsqueeze(1)
        for idx, conv in enumerate(self.conv_layers_list):
            # conv with batch norm next → no relu. conv without batch norm next → relu. batch norm → relu (falls through to last line). pool → no relu
            if isinstance(conv, nn.MaxPool2d) or _is_next_batch_norm(
                self.conv_layers_list, idx
            ):
                x = conv(x)
                continue
            x = self.activation_fn(conv(x))

        if self.global_avg_pool is not None:
            device = x.device
            x = self.global_avg_pool(x.cpu()).to(device)

        x = torch.flatten(x, 1)

        for fc in self.fc_layers_list[:-1]:
            if isinstance(fc, nn.Dropout):
                x = fc(x)
                continue
            x = self.activation_fn(fc(x))

        x = self.fc_layers_list[-1](x)

        return x
