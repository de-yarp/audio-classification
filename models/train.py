from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

from infra.data_models import (
    OPTIMIZER_MAP,
    AudioDataset,
    ConfigCNN,
    ConfigLSTM,
    ModelType,
    ReprType,
)

from .cnn_mel import MEL_CNN
from .cnn_mfcc import MFCC_CNN
from .lstm_mel import MEL_LSTM
from .lstm_mfcc import MFCC_LSTM


def _get_model_type_from_yaml(path: Path) -> tuple[ModelType, ReprType]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["model"]
    model_type_str = model_cfg["model_type"].lower()
    repr_type_str = model_cfg["repr_type"].lower()

    try:
        model_type = ModelType(model_type_str)
    except ValueError as e:
        msg = f"invalid model.type '{model_type_str}' in config {path}, expected {[e.value for e in ModelType]}"
        raise ValueError(msg) from e

    try:
        repr_type = ReprType(repr_type_str)
    except ValueError as e:
        msg = f"invalid model.repr_type '{repr_type_str}' in config {path}, expected {[e.value for e in ReprType]}"
        raise ValueError(msg) from e

    return model_type, repr_type


def _setup_optimizer(
    net: nn.Module, cfg: ConfigCNN | ConfigLSTM, cfg_path: Path
) -> optim.Optimizer:
    opt_class = OPTIMIZER_MAP.get(cfg.optimizer, None)
    if opt_class is None:
        msg = f"invalid run.optimizer '{cfg.optimizer}' in config {cfg_path}, expected {OPTIMIZER_MAP.keys()}"
        raise ValueError(msg)

    if opt_class is optim.SGD and cfg.momentum is not None:
        return opt_class(params=net.parameters(), lr=cfg.lr, momentum=cfg.momentum)

    return opt_class(params=net.parameters(), lr=cfg.lr)


def _setup_model(cfg_path: Path) -> tuple[nn.Module, ConfigCNN | ConfigLSTM]:
    model_type, repr_type = _get_model_type_from_yaml(cfg_path)

    if model_type == ModelType.CNN:
        cfg = ConfigCNN.from_yaml(cfg_path)
        if repr_type == ReprType.MFCC:
            net = MFCC_CNN(cfg=cfg)
        elif repr_type == ReprType.MEL:
            net = MEL_CNN(cfg=cfg)

    elif model_type == ModelType.LSTM:
        cfg = ConfigLSTM.from_yaml(cfg_path)
        if repr_type == ReprType.MFCC:
            net = MFCC_LSTM(cfg=cfg)
        elif repr_type == ReprType.MEL:
            net = MEL_LSTM(cfg=cfg)

    return net, cfg


def run_validation(
    net: nn.Module, val_loader: DataLoader, criterion: nn.CrossEntropyLoss
) -> None:
    net.eval()
    val_loss = 0.0

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = net(inputs)
            predicted = torch.argmax(outputs, 1)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy_pct = (correct / total) * 100
    avg_loss = val_loss / len(val_loader)

    print(f"Validation loss: {avg_loss:.3f}, accuracy: {accuracy_pct:.3f}")


def training_loop(cfg_path: Path):
    net, cfg = _setup_model(cfg_path)

    criterion = nn.CrossEntropyLoss()
    optimizer = _setup_optimizer(net, cfg, cfg_path)

    train_ds = AudioDataset(cfg.repr_type, folds=cfg.folds_train)
    val_ds = AudioDataset(cfg.repr_type, folds=cfg.folds_val)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    for epoch in range(cfg.num_epochs):
        running_loss = 0.0

        net.train()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss:.3f}")
            running_loss = 0.0

        run_validation(net, val_loader, criterion)

    print("Finished Training")
