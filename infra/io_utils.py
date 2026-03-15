import csv
import datetime as dt
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import yaml

from .data_models import (
    MODEL_CHECKPOINTS_DIR_PATH,
    MODEL_CONFIGS_DIR_PATH,
    ConfigCNN,
    ConfigLSTM,
    ModelType,
    ReprType,
)

COMPONENT = __name__


def now_ts_str_filename() -> str:
    return dt.datetime.now(tz=dt.timezone.utc).strftime("%Y%m%d_%H%M%S")


def save_features_npy(
    mel_spec: np.ndarray, mfcc: np.ndarray, out_dir: Path, file_name: str
) -> None:
    mel_dir = out_dir / "mel"
    mfcc_dir = out_dir / "mfcc"
    mel_dir.mkdir(exist_ok=True, parents=True)
    mfcc_dir.mkdir(exist_ok=True, parents=True)

    # fout_mel = mel_dir / f"mel_{file_name}.npy"
    # fout_mfcc = mfcc_dir / f"mfcc_{file_name}.npy"

    fout_mel = mel_dir / f"{file_name}.npy"
    fout_mfcc = mfcc_dir / f"{file_name}.npy"

    np.save(fout_mel, mel_spec)
    np.save(fout_mfcc, mfcc)


def load_yaml_config(path: Path) -> dict:
    def _flatten(cfg_raw: dict) -> dict:
        return cfg_raw["model"] | cfg_raw["run"]

    with path.open("r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)

    cfg_dict_flat = _flatten(cfg_dict)

    return cfg_dict_flat


def save_yaml_config(
    cfg_instance: ConfigCNN | ConfigLSTM,
    model_type: ModelType,
    repr_type: ReprType,
    ts_filename_ext: str,
    *,
    path: Path = MODEL_CONFIGS_DIR_PATH,
    emit: Callable[[str, str, str, dict], None],
) -> Path:
    cfg = cfg_instance.to_dict()
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError
    save_path = path / f"{model_type.value}_{repr_type.value}_{ts_filename_ext}.yaml"
    with save_path.open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f)

    emit(
        level="INFO",
        component=COMPONENT,
        event="save_model_config",
        payload={"cfg_path": str(save_path)},
    )

    return save_path


def save_model_checkpoint(
    net: torch.nn.Module,
    model_type: ModelType,
    repr_type: ReprType,
    ts_filename_ext: str,
    *,
    path: Path = MODEL_CHECKPOINTS_DIR_PATH,
    emit: Callable[[str, str, str, dict], None],
) -> Path:
    save_path = path / f"{model_type.value}_{repr_type.value}_{ts_filename_ext}.pt"
    torch.save(
        net.state_dict(),
        save_path,
    )

    emit(
        level="INFO",
        component=COMPONENT,
        event="save_model_checkpoint",
        payload={"model_path": str(save_path)},
    )

    return save_path


def write_data_to_csv(
    content: dict, csv_path: Path, *, emit: Callable[[str, str, str, dict], None]
) -> None:
    fieldnames = [
        "ts",
        "run_id",
        "avg_loss_last_train_epoch",
        "avg_loss_val",
        "accuracy_val_pct",
        "cfg_path",
        "model_path",
    ]

    with csv_path.open("a+", encoding="utf-8") as f:
        csv_writer = csv.DictWriter(
            f, fieldnames=fieldnames, escapechar="\\", quoting=csv.QUOTE_MINIMAL
        )
        if csv_path.stat().st_size == 0:
            csv_writer.writeheader()
        csv_writer.writerow(content)

    emit(
        level="INFO",
        component=COMPONENT,
        event="write_run_info_to_csv",
        payload={"csv_path": str(csv_path)},
    )


def save_run_info(
    save_model: bool,
    net: torch.nn.Module,
    cfg_instance: ConfigCNN | ConfigLSTM,
    cfg_path: Path,
    content: dict,
    csv_path: Path,
    *,
    emit: Callable[[str, str, str, dict], None],
) -> None:
    content_updated = content.copy()
    ts_filename_ext = now_ts_str_filename()
    model_path = None
    cfg_path = save_yaml_config(
        cfg_instance,
        cfg_instance.model_type,
        cfg_instance.repr_type,
        ts_filename_ext,
        emit=emit,
    )

    if save_model:
        model_path = save_model_checkpoint(
            net,
            cfg_instance.model_type,
            cfg_instance.repr_type,
            ts_filename_ext,
            emit=emit,
        )

    content_updated["cfg_path"] = cfg_path
    content_updated["model_path"] = model_path

    write_data_to_csv(content_updated, csv_path, emit=emit)
