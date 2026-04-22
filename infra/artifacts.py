import json
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

COMPONENT = __name__


def compute_confusion_matrix(
    preds: np.ndarray,
    labels: np.ndarray,
    *,
    normalize: str | None = None,
    category_level: bool = False,
) -> np.ndarray:
    if category_level:
        labels = labels // 10
        preds = preds // 10
    return confusion_matrix(labels, preds, normalize=normalize)


def save_confusion_matrix(
    preds: np.ndarray,
    labels: np.ndarray,
    run_id: str,
    output_dir: Path,
    class_names: list[str] | None = None,
    *,
    emit: Callable[[str, str, str, dict], None],
    precomputed_cm: np.ndarray | None = None,
    custom_stem: str | None = None,
) -> tuple[Path, Path]:

    if precomputed_cm is None:
        cm = confusion_matrix(labels, preds)
    else:
        cm = precomputed_cm

    if custom_stem is None:
        file_stem = "confusion_matrix"
    else:
        file_stem = custom_stem

    npy_path = output_dir / (file_stem + ".npy")
    np.save(npy_path, cm)

    emit(
        level="INFO",
        component=COMPONENT,
        event="save_artifact_confusion_matrix_npy",
        payload={"artifact_path": str(npy_path)},
    )

    fmt = "d" if cm.dtype.kind == "i" else ".2f"
    annot = cm.shape[0] <= 10  # annotate only small matrices

    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(
        cm,
        annot=annot,
        fmt=fmt if annot else "",
        cmap="Blues",
        xticklabels=class_names or range(cm.shape[1]),
        yticklabels=class_names or range(cm.shape[0]),
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {run_id}")
    plt.tight_layout()

    png_path = output_dir / (file_stem + ".png")
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


# --- add to artifacts.py ---


def save_cv_loss_curve(
    epochs: range,
    train_mean: np.ndarray,
    train_std: np.ndarray,
    val_mean: np.ndarray,
    val_std: np.ndarray,
    cv_run_id: str,
    output_dir: Path,
    *,
    emit: Callable[[str, str, str, dict], None],
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_mean, label="Train Loss (mean)")
    ax.fill_between(epochs, train_mean - train_std, train_mean + train_std, alpha=0.2)
    ax.plot(epochs, val_mean, label="Val Loss (mean)")
    ax.fill_between(epochs, val_mean - val_std, val_mean + val_std, alpha=0.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"CV Loss Curve — {cv_run_id}")
    ax.legend()
    plt.tight_layout()

    path = output_dir / "cv_loss_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    emit(
        level="INFO",
        component=COMPONENT,
        event="save_artifact_cv_loss_curve_png",
        payload={"artifact_path": str(path)},
    )

    return path


def save_cv_accuracy_curve(
    epochs: range,
    acc_mean: np.ndarray,
    acc_std: np.ndarray,
    cv_run_id: str,
    output_dir: Path,
    *,
    emit: Callable[[str, str, str, dict], None],
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, acc_mean, label="Val Accuracy (mean)", color="green")
    ax.fill_between(
        epochs, acc_mean - acc_std, acc_mean + acc_std, alpha=0.2, color="green"
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"CV Validation Accuracy — {cv_run_id}")
    ax.legend()
    plt.tight_layout()

    path = output_dir / "cv_accuracy_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    emit(
        level="INFO",
        component=COMPONENT,
        event="save_artifact_cv_accuracy_curve_png",
        payload={"artifact_path": str(path)},
    )

    return path
