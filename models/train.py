from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from infra.data_models import (
    OPTIMIZER_MAP,
    ArgsCLI,
    AudioDataset,
    ConfigCNN,
    ConfigLSTM,
    ModelType,
    ReprType,
)
from infra.log_utils import now_ts_iso

from .cnn_mel import MEL_CNN
from .cnn_mfcc import MFCC_CNN
from .lstm_mel import MEL_LSTM
from .lstm_mfcc import MFCC_LSTM

COMPONENT = __name__


def _setup_optimizer(
    net: nn.Module, cfg: ConfigCNN | ConfigLSTM, cfg_path: Path
) -> optim.Optimizer:
    opt_class = OPTIMIZER_MAP.get(cfg.optimizer, None)
    if opt_class is None:
        msg = f"invalid run.optimizer '{cfg.optimizer}' in config {cfg_path}, expected {OPTIMIZER_MAP.keys()}"
        raise ValueError(msg)

    if opt_class is optim.SGD and cfg.momentum is not None:
        return opt_class(
            params=net.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )

    return opt_class(params=net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


def _setup_model(
    cfg_dict: dict, model_path: Path | None = None
) -> tuple[nn.Module, ConfigCNN | ConfigLSTM, torch.device]:
    model_type = cfg_dict["model_type"]
    repr_type = cfg_dict["repr_type"]

    if model_type == ModelType.CNN:
        cfg = ConfigCNN.from_dict(cfg_dict)
        if repr_type == ReprType.MFCC:
            net = MFCC_CNN(cfg=cfg)
        elif repr_type == ReprType.MEL:
            net = MEL_CNN(cfg=cfg)

    elif model_type == ModelType.LSTM:
        cfg = ConfigLSTM.from_dict(cfg_dict)
        if repr_type == ReprType.MFCC:
            net = MFCC_LSTM(cfg=cfg)
        elif repr_type == ReprType.MEL:
            net = MEL_LSTM(cfg=cfg)

    if model_path is not None:
        net.load_state_dict(torch.load(model_path))

    device = None

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    net.to(device)

    return net, cfg, device


def run_validation(
    net: nn.Module,
    val_loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    emit: Callable[[str, str, str, dict], None],
) -> tuple[float, float]:
    net.eval()
    val_loss = 0.0

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = net(inputs)
            predicted = torch.argmax(outputs, 1)

            loss = criterion(outputs, labels)
            val_loss += loss.item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy_pct = (correct / total) * 100
    avg_loss = val_loss / len(val_loader)

    emit(
        level="INFO",
        component=COMPONENT,
        event="validation_result",
        payload={
            "avg_loss": round(avg_loss, 4),
            "accuracy_pct": round(accuracy_pct, 4),
        },
    )

    return avg_loss, accuracy_pct


def training_loop(
    cfg_dict: dict,
    *,
    emit: Callable[[str, str, str, dict], None],
    run_id: str,
    cv_run_id: str | None = None,
    args: ArgsCLI,
) -> tuple[nn.Module, ConfigCNN | ConfigLSTM, dict, dict]:

    emit(
        level="INFO",
        component=COMPONENT,
        event="start_training",
        payload={"parent_cv_run_id": cv_run_id},
    )

    net, cfg, device = _setup_model(cfg_dict)

    criterion = nn.CrossEntropyLoss()
    optimizer = _setup_optimizer(net, cfg, args.cfg_path)

    train_ds = AudioDataset(cfg.repr_type, folds=cfg.folds_train)
    val_ds = AudioDataset(cfg.repr_type, folds=cfg.folds_val)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    avg_loss_last_train_epoch = 0.0
    avg_loss_val = 0.0
    accuracy_val_pct = 0.0

    train_losses = []
    val_losses = []
    val_accuracies = []
    for epoch in range(cfg.num_epochs):
        running_loss = 0.0

        net.train()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            emit(
                level="INFO",
                component=COMPONENT,
                event="batch_loss",
                payload={
                    "epoch": epoch + 1,
                    "batch": i + 1,
                    "loss": round(loss.item(), 4),
                },
            )

        avg_loss_last_train_epoch = running_loss / len(train_loader)

        avg_loss_val, accuracy_val_pct = run_validation(
            net, val_loader, criterion, device, emit
        )

        train_losses.append(avg_loss_last_train_epoch)
        val_losses.append(avg_loss_val)
        val_accuracies.append(accuracy_val_pct)

    emit(
        level="INFO",
        component=COMPONENT,
        event="finish_training",
        payload={
            "avg_loss_last_train_epoch": round(avg_loss_last_train_epoch, 4),
            "avg_loss_val": round(avg_loss_val, 4),
            "accuracy_val_pct": round(accuracy_val_pct, 4),
            "parent_cv_run_id": cv_run_id,
        },
    )

    content = {
        "ts": now_ts_iso(),
        "run_id": run_id,
        "avg_loss_last_train_epoch": round(avg_loss_last_train_epoch, 4),
        "avg_loss_val": round(avg_loss_val, 4),
        "accuracy_val_pct": round(accuracy_val_pct, 4),
    }

    return (
        net,
        cfg,
        content,
        {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_accuracies": val_accuracies,
        },
    )
