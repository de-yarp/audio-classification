from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from infra.data_models import (
    OPTIMIZER_MAP,
    SCHEDULER_MAP,
    ArgsCLI,
    AudioDataset,
    CMInfo,
    ConfigCNN,
    ConfigLSTM,
    DatasetType,
    ModelType,
    ReprType,
    SchedulerType,
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
    optimizer_type = cfg.optimizer
    opt_class = OPTIMIZER_MAP.get(optimizer_type.value, None)
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


def _setup_scheduler(
    optimizer: optim.Optimizer, cfg: ConfigCNN | ConfigLSTM, cfg_path: Path
) -> optim.lr_scheduler.LRScheduler | None:
    scheduler_type = cfg.scheduler
    if scheduler_type is None:
        return None

    scheduler_class = SCHEDULER_MAP.get(scheduler_type.value, None)
    if scheduler_type != SchedulerType.COSINE:
        assert cfg.factor is not None, (
            f"invalid run.factor '{cfg.factor}' in config {cfg_path}, for expected scheduler type '{scheduler_type.value}'"
        )
    if scheduler_type == SchedulerType.PLATEAU:
        assert cfg.patience is not None, (
            f"invalid run.patience '{cfg.patience}' in config {cfg_path}, for expected scheduler type '{scheduler_type.value}'"
        )
    if scheduler_type == SchedulerType.STEP:
        assert cfg.step_size is not None, (
            f"invalid run.step_size '{cfg.step_size}' in config {cfg_path}, for expected scheduler type '{scheduler_type.value}'"
        )
    assert cfg.min_lr is not None, (
        f"invalid run.min_lr '{cfg.min_lr}' in config {cfg_path}, for expected scheduler type '{scheduler_type.value}'"
    )

    if scheduler_type == SchedulerType.PLATEAU:
        return scheduler_class(
            optimizer, factor=cfg.factor, patience=cfg.patience, min_lr=cfg.min_lr
        )
    elif scheduler_type == SchedulerType.COSINE:
        return scheduler_class(optimizer, T_max=cfg.num_epochs, eta_min=cfg.min_lr)
    elif scheduler_type == SchedulerType.STEP:
        return scheduler_class(optimizer, step_size=cfg.step_size, gamma=cfg.factor)
    else:
        return None


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
    class_names: list[str],
    criterion: nn.CrossEntropyLoss,
    current_lr: float,
    device: torch.device,
    emit: Callable[[str, str, str, dict], None],
) -> tuple[float, float, CMInfo]:
    net.eval()
    val_loss = 0.0

    all_preds: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = net(inputs)
            predicted = torch.argmax(outputs, 1)

            all_preds.append(predicted)
            all_labels.append(labels)

            loss = criterion(outputs, labels)
            val_loss += loss.item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy_pct = (correct / total) * 100
    avg_loss = val_loss / len(val_loader)

    all_preds_flat = torch.cat(all_preds).to(device="cpu").numpy()
    all_labels_flat = torch.cat(all_labels).to(device="cpu").numpy()

    emit(
        level="INFO",
        component=COMPONENT,
        event="validation_result",
        payload={
            "avg_loss": round(avg_loss, 4),
            "accuracy_pct": round(accuracy_pct, 4),
            "lr": current_lr,
        },
    )

    cm_info: CMInfo = {
        "preds": all_preds_flat,
        "labels": all_labels_flat,
        "class_names": class_names,
    }

    return avg_loss, accuracy_pct, cm_info


def training_loop(
    cfg_dict: dict,
    *,
    emit: Callable[[str, str, str, dict], None],
    run_id: str,
    cv_run_id: str | None = None,
    args: ArgsCLI,
) -> tuple[nn.Module, ConfigCNN | ConfigLSTM, dict, dict, CMInfo]:

    emit(
        level="INFO",
        component=COMPONENT,
        event="start_training",
        payload={"parent_cv_run_id": cv_run_id},
    )

    net, cfg, device = _setup_model(cfg_dict)
    print(
        f"[TRAIN] Training started | epochs={cfg.num_epochs} | device={device} | run_id={run_id}"
    )

    if isinstance(cfg, ConfigLSTM):
        emit(
            level="INFO",
            component=COMPONENT,
            event="model_init",
            payload={"bidirectional": cfg.bidirectional, "pooling": cfg.pooling},
        )

    criterion = nn.CrossEntropyLoss()
    optimizer = _setup_optimizer(net, cfg, args.cfg_path)
    scheduler = _setup_scheduler(optimizer, cfg, args.cfg_path)

    train_ds = AudioDataset(
        repr_type=cfg.repr_type,
        folds=cfg.folds_train,
        dataset_type=DatasetType.TRAIN,
        cfg=cfg,
    )
    val_ds = AudioDataset(
        cfg.repr_type, folds=cfg.folds_val, dataset_type=DatasetType.VAL
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    avg_loss_last_train_epoch = 0.0
    avg_loss_val = 0.0
    accuracy_val_pct = 0.0

    train_losses = []
    val_losses = []
    val_accuracies = []
    last_val_info_for_cms: CMInfo | None = None

    for epoch in range(cfg.num_epochs):
        running_loss = 0.0

        net.train()

        if cfg.warmup_lr:
            warmup_epochs = cfg.warmup_epochs
            warmup_lr = cfg.warmup_lr_val
            lr = cfg.lr
            if epoch < warmup_epochs:
                warmup_factor = (epoch + 1) / warmup_epochs
                current_lr = warmup_lr + (lr - warmup_lr) * warmup_factor
                for pg in optimizer.param_groups:
                    pg["lr"] = current_lr

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

        avg_loss_val, accuracy_val_pct, cm_info = run_validation(
            net,
            val_loader,
            val_ds.class_names,
            criterion,
            optimizer.param_groups[0]["lr"],
            device,
            emit,
        )
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"  epoch {epoch + 1:>3}/{cfg.num_epochs}"
            f" | train_loss={avg_loss_last_train_epoch:.4f}"
            f" | val_loss={avg_loss_val:.4f}"
            f" | val_acc={accuracy_val_pct:.2f}%"
            f" | lr={current_lr:.6f}"
        )

        if epoch == cfg.num_epochs - 1:
            last_val_info_for_cms = cm_info

        train_losses.append(avg_loss_last_train_epoch)
        val_losses.append(avg_loss_val)
        val_accuracies.append(accuracy_val_pct)

        if scheduler is not None:
            if not cfg.warmup_lr or epoch >= cfg.warmup_epochs:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(avg_loss_val)
                else:
                    scheduler.step()

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
    print(
        f"[TRAIN] Finished | val_acc={accuracy_val_pct:.2f}% | val_loss={avg_loss_val:.4f}"
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
        last_val_info_for_cms,
    )
