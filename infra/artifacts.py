import json
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

COMPONENT = __name__


def save_confusion_matrix(
    preds: np.ndarray,
    labels: np.ndarray,
    run_id: str,
    output_dir: Path,
    class_names: list[str] | None = None,
    *,
    emit: Callable[[str, str, str, dict], None],
) -> tuple[Path, Path]:
    cm = confusion_matrix(labels, preds)
    npy_path = output_dir / "confusion_matrix.npy"
    np.save(npy_path, cm)

    emit(
        level="INFO",
        component=COMPONENT,
        event="save_artifact_confusion_matrix_npy",
        payload={"artifact_path": str(npy_path)},
    )

    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names or range(cm.shape[1]),
        yticklabels=class_names or range(cm.shape[0]),
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {run_id}")
    plt.tight_layout()

    png_path = output_dir / "confusion_matrix.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    emit(
        level="INFO",
        component=COMPONENT,
        event="save_artifact_confusion_matrix_png",
        payload={"artifact_path": str(png_path)},
    )

    return npy_path, png_path


def save_classification_report(
    preds: np.ndarray,
    labels: np.ndarray,
    run_id: str,
    output_dir: Path,
    class_names: list[str] | None = None,
    *,
    emit: Callable[[str, str, str, dict], None],
) -> Path:
    report = classification_report(
        labels,
        preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    path = output_dir / "classification_report.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2)

    emit(
        level="INFO",
        component=COMPONENT,
        event="save_artifact_classification_report_json",
        payload={"artifact_path": str(path)},
    )

    return path


def save_loss_curve(
    train_losses: list[float],
    val_losses: list[float],
    run_id: str,
    output_dir: Path,
    *,
    emit: Callable[[str, str, str, dict], None],
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train Loss")
    ax.plot(epochs, val_losses, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Loss Curve — {run_id}")
    ax.legend()
    plt.tight_layout()

    path = output_dir / "loss_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    emit(
        level="INFO",
        component=COMPONENT,
        event="save_artifact_loss_curve_png",
        payload={"artifact_path": str(path)},
    )

    return path


def save_accuracy_curve(
    val_accuracies: list[float],
    run_id: str,
    output_dir: Path,
    *,
    emit: Callable[[str, str, str, dict], None],
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(val_accuracies) + 1)
    ax.plot(epochs, val_accuracies, label="Val Accuracy", color="green")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Validation Accuracy — {run_id}")
    ax.legend()
    plt.tight_layout()

    path = output_dir / "accuracy_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    emit(
        level="INFO",
        component=COMPONENT,
        event="save_artifact_accuracy_curve_png",
        payload={"artifact_path": str(path)},
    )

    return path
