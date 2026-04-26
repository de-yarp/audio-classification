from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader

from infra.data_models import ArgsCLI, AudioDataset, DatasetType
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
    print(f"[EVAL] Evaluation started | run_id={run_id}")

    net, cfg, device = _setup_model(cfg_dict, args.model_path)

    net.eval()
    criterion = nn.CrossEntropyLoss()
    eval_loss = 0.0

    eval_ds = AudioDataset(
        repr_type=cfg.repr_type, folds=args.eval_folds, dataset_type=DatasetType.EVAL
    )
    eval_loader = DataLoader(eval_ds, batch_size=cfg.batch_size)

    all_preds: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in eval_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

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

    all_preds_flat = torch.cat(all_preds).to(device="cpu").numpy()
    all_labels_flat = torch.cat(all_labels).to(device="cpu").numpy()

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

    # В кінці функції evaluate_model, перед поверненням значень:
    all_preds_np = torch.cat(all_preds).cpu().numpy()
    all_labels_np = torch.cat(all_labels).cpu().numpy()

    # Обчислюємо Precision, Recall та F1
    p, r, f1, _ = precision_recall_fscore_support(
        all_labels_np, all_preds_np, average="macro", zero_division=0
    )

    emit(
        level="INFO",
        component=COMPONENT,
        event="finish_eval",
        payload={
            "avg_loss": round(eval_loss / len(eval_loader), 4),
            "accuracy_pct": round((correct / total) * 100, 4),
            "precision_macro": round(p * 100, 4),
            "recall_macro": round(r * 100, 4),
            "f1_macro": round(f1 * 100, 4),
        },
    )
    print(
        f"[EVAL] Done"
        f" | acc={accuracy_pct:.2f}%"
        f" | loss={avg_loss:.4f}"
        f" | precision={p * 100:.2f}%"
        f" | recall={r * 100:.2f}%"
        f" | f1={f1 * 100:.2f}%"
    )
    return all_preds_flat, all_labels_flat, eval_ds.class_names, content
