from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from infra.data_models import (
    ArgsCLI,
    AudioDataset,
)
from infra.log_utils import now_ts_iso

from .train import _setup_model

COMPONENT = __name__


def evaluate_model(
    cfg_dict: dict,
    *,
    emit: Callable[[str, str, str, dict], None],
    run_id: str,
    args: ArgsCLI,
) -> tuple[np.ndarray, np.ndarray, list[str], dict]:

    emit(
        level="INFO",
        component=COMPONENT,
        event="start_eval",
    )

    net, cfg = _setup_model(cfg_dict, args.model_path)

    net.eval()
    criterion = nn.CrossEntropyLoss()
    eval_loss = 0.0

    eval_ds = AudioDataset(cfg.repr_type, args.eval_folds)
    eval_loader = DataLoader(eval_ds, batch_size=cfg.batch_size)

    all_preds: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in eval_loader:
            outputs = net(inputs)
            predicted = torch.argmax(outputs, 1)
            all_preds.append(predicted)
            all_labels.append(labels)
            loss = criterion(outputs, labels)
            eval_loss += loss.item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = eval_loss / len(eval_loader)
    accuracy_pct = (correct / total) * 100

    all_preds_flat = torch.cat(all_preds).numpy()
    all_labels_flat = torch.cat(all_labels).numpy()

    emit(
        level="INFO",
        component=COMPONENT,
        event="finish_eval",
        payload={
            "avg_loss": round(avg_loss, 4),
            "accuracy_pct": round(accuracy_pct, 4),
        },
    )

    content = {
        "ts": now_ts_iso(),
        "run_id": run_id,
        "avg_loss": round(avg_loss, 4),
        "accuracy_pct": round(accuracy_pct, 4),
    }

    return all_preds_flat, all_labels_flat, eval_ds.class_names, content
