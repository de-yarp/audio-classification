import math
from dataclasses import asdict, dataclass, fields
from enum import Enum
from pathlib import Path
from typing import Callable, TypedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml


@dataclass(frozen=True)
class FeatureConfig:
    n_fft: int
    hop_length: int
    n_mels: int
    n_mfcc: int
    normalize_mfcc: bool = False
    include_deltas: bool = False
    stack_deltas_as_channels: bool = False

    @classmethod
    def from_yaml(cls, path: Path):
        with path.open(mode="r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        inner_keys = {f.name for f in fields(cls)}
        data = {k: v for k, v in cfg.items() if k in inner_keys}

        return cls(**data)


class ReprType(Enum):
    MEL = "mel"
    MFCC = "mfcc"


class ModelType(Enum):
    CNN = "cnn"
    LSTM = "lstm"


class CNNLayers(Enum):
    CONV = "conv"
    POOL = "pool"


class PoolType(Enum):
    MAX = "max"
    AVG = "avg"


class SchedulerType(Enum):
    PLATEAU = "PLATEAU"
    COSINE = "COSINE"
    STEP = "STEP"


class OptimizerType(Enum):
    SGD = "SGD"
    ADAM = "ADAM"
    ADAMW = "ADAMW"


class DatasetType(Enum):
    TRAIN = "train"
    VAL = "val"
    EVAL = "eval"


ESC_50_RAW_PATH = Path("data") / "raw" / "esc50"
ESC_50_PROCESSED_PATH = Path("data") / "processed" / "esc50"
LOGS_DIR_PATH = Path("logs")
LOG_NAME = "log.jsonl"
MODEL_CHECKPOINTS_DIR_PATH = Path("runs") / "checkpoints"
MODEL_CONFIGS_DIR_PATH = Path("runs") / "configs"
MODEL_ARTIFACTS_DIR_PATH = Path("runs") / "artifacts"


@dataclass(frozen=True)
class LayerConv:
    kernel_count: int
    kernel_size: int | list[int]
    stride: int
    padding: int | list[int]
    batch_norm: bool = False
    type: str = "conv"


@dataclass(frozen=True)
class LayerPool:
    kernel_size: int | list[int]
    stride: int | list[int]
    padding: int
    type: str = "pool"


@dataclass(frozen=True)
class ConfigCNN:
    # model config
    model_type: ModelType
    repr_type: ReprType
    conv_layers: list[LayerConv | LayerPool]
    pool_type: PoolType
    fc_layers: list[int]
    dropout: float
    num_classes: int
    global_avg_pool: list[int] | None

    # run config
    seed: int
    batch_size: int
    folds_train: list[int]
    folds_val: list[int]
    num_epochs: int

    optimizer: OptimizerType
    lr: float
    momentum: float | None
    weight_decay: float

    warmup_lr: bool
    warmup_epochs: int | None
    warmup_lr_val: float | None

    scheduler: SchedulerType | None
    factor: float | None
    patience: int | None
    min_lr: float | None
    step_size: int | None

    # data augmentation
    augment: bool
    freq_masks: int | None
    freq_mask_width: int | None
    time_masks: int | None
    time_mask_width: int | None

    # in_channels switch for mfcc
    mfcc_deltas: bool = True
    stack_deltas_as_channels: bool = False

    @classmethod
    def from_dict(cls, input: dict):
        inner_keys = {f.name for f in fields(cls)}
        data = {k: v for k, v in input.items() if k in inner_keys}

        return cls(**data)

    def to_dict(self) -> dict:
        model_keys = {
            "model_type",
            "repr_type",
            "conv_layers",
            "fc_layers",
            "num_classes",
            "mfcc_deltas",
            "stack_deltas_as_channels",
            "pool_type",
            "dropout",
            "global_avg_pool",
        }
        run_keys = {
            "seed",
            "batch_size",
            "folds_train",
            "folds_val",
            "num_epochs",
            "optimizer",
            "lr",
            "momentum",
            "weight_decay",
            "warmup_lr",
            "warmup_epochs",
            "warmup_lr_val",
            "scheduler",
            "factor",
            "patience",
            "min_lr",
            "step_size",
            "augment",
            "freq_masks",
            "freq_mask_width",
            "time_masks",
            "time_mask_width",
        }
        raw_dict = asdict(self)
        model_dict = {}
        run_dict = {}
        for k, v in raw_dict.items():
            if k in model_keys:
                if isinstance(v, Enum):
                    model_dict[k] = v.value
                    continue
                model_dict[k] = v
            if k in run_keys:
                if isinstance(v, Enum):
                    run_dict[k] = v.value
                    continue
                run_dict[k] = v

        return {"model": model_dict, "run": run_dict}


@dataclass(frozen=True)
class ConfigLSTM:
    # model config
    model_type: ModelType
    repr_type: ReprType
    hidden_size: int
    num_layers: int
    dropout: float
    fc_layers: list[int]
    num_classes: int

    # run config
    seed: int
    batch_size: int
    folds_train: list[int]
    folds_val: list[int]
    num_epochs: int

    optimizer: OptimizerType
    lr: float
    momentum: float | None
    weight_decay: float

    warmup_lr: bool
    warmup_epochs: int | None
    warmup_lr_val: float | None

    scheduler: SchedulerType | None
    factor: float | None
    patience: int | None
    min_lr: float | None
    step_size: int | None

    # data augmentation
    augment: bool
    freq_masks: int | None
    freq_mask_width: int | None
    time_masks: int | None
    time_mask_width: int | None

    # in_channels switch for mfcc
    mfcc_deltas: bool = True
    bidirectional: bool = False
    pooling: str = "last"

    @classmethod
    def from_dict(cls, input: dict):
        inner_keys = {f.name for f in fields(cls)}
        data = {k: v for k, v in input.items() if k in inner_keys}

        return cls(**data)

    def to_dict(self) -> dict:
        model_keys = {
            "model_type",
            "repr_type",
            "mfcc_deltas",
            "hidden_size",
            "num_layers",
            "dropout",
            "fc_layers",
            "num_classes",
            "bidirectional",
            "pooling",
        }
        run_keys = {
            "seed",
            "batch_size",
            "folds_train",
            "folds_val",
            "num_epochs",
            "optimizer",
            "lr",
            "momentum",
            "weight_decay",
            "warmup_lr",
            "warmup_epochs",
            "warmup_lr_val",
            "scheduler",
            "factor",
            "patience",
            "min_lr",
            "step_size",
            "augment",
            "freq_masks",
            "freq_mask_width",
            "time_masks",
            "time_mask_width",
        }
        raw_dict = asdict(self)
        model_dict = {}
        run_dict = {}
        for k, v in raw_dict.items():
            if k in model_keys:
                if isinstance(v, Enum):
                    model_dict[k] = v.value
                    continue
                model_dict[k] = v
            if k in run_keys:
                if isinstance(v, Enum):
                    run_dict[k] = v.value
                    continue
                run_dict[k] = v

        return {"model": model_dict, "run": run_dict}


class AudioDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        repr_type: ReprType,
        folds: list[int],
        dataset_type: DatasetType,
        *,
        cfg: ConfigCNN | ConfigLSTM | None = None,
        root_dir: Path = ESC_50_PROCESSED_PATH,
        csv_file: Path = ESC_50_RAW_PATH / "meta" / "esc50.csv",
    ):
        self.dataset_type = dataset_type
        self.folds = folds
        self.repr_dir = root_dir / repr_type.value
        self.cfg = cfg
        self.meta_df = pd.read_csv(csv_file, delimiter=",")
        self.class_names = self._get_class_names()
        self.available_folds = self.meta_df["fold"].unique()
        self._validate_input_folds()

        self.samples, self.labels = self._load_samples()

        self.shape = self._get_shape()

        self._validate_augment()

    def _load_samples(self) -> tuple[list[np.ndarray], list[int]]:
        def _get_path(x: str) -> Path:
            p = Path(x).with_suffix(".npy")
            return self.repr_dir / p

        folds_mask = self.meta_df["fold"].isin(self.folds)
        df = self.meta_df[folds_mask]

        labels = df["target"].to_list()
        samples_paths = df["filename"].map(_get_path)
        samples = [np.load(p) for p in samples_paths]

        return samples, labels

    def _validate_input_folds(self) -> None:
        for fold in self.folds:
            assert np.any(self.available_folds == fold), (
                f"could not init AudioDataset class: input folds {self.folds} contain invalid folds, available folds={self.available_folds}"
            )

    def _get_class_names(self) -> list[str]:
        class_names_indexed = self.meta_df[["target", "category"]].sort_values(
            by=["target"]
        )
        return class_names_indexed["category"].unique().tolist()

    def _get_shape(self) -> tuple[int, int] | tuple[int, int, int]:
        return self.samples[0].shape

    def _validate_augment(self) -> None:
        if self.dataset_type == DatasetType.TRAIN and self.cfg is None:
            msg = f"dataset_type='{self.dataset_type}', but cfg is None; expected cfg of type ConfigCNN | ConfigLSTM."
            raise TypeError(msg)

        max_signal_override_perc = 50

        if (
            self.dataset_type == DatasetType.TRAIN
            and self.cfg is not None
            and self.cfg.augment
        ):
            freq_mask_width = self.cfg.freq_mask_width
            freq_masks = self.cfg.freq_masks
            time_mask_width = self.cfg.time_mask_width
            time_masks = self.cfg.time_masks

            assert freq_mask_width >= 1, (
                f"expected freq_mask_width >= 1, received cfg.run.freq_mask_width={freq_mask_width}"
            )
            assert freq_masks >= 0, (
                f"expected freq_masks >= 0, received cfg.run.freq_masks={freq_masks}"
            )
            assert time_mask_width >= 1, (
                f"expected time_mask_width >= 1, received cfg.run.time_mask_width={time_mask_width}"
            )
            assert time_masks >= 1, (
                f"expected time_masks >= 1, received cfg.run.time_masks={time_masks}"
            )

            freq_axis = -2
            time_axis = -1

            max_freq_mask_perc = math.ceil(
                ((freq_mask_width * freq_masks) / self.shape[freq_axis]) * 100
            )
            max_time_mask_perc = math.ceil(
                ((time_mask_width * time_masks) / self.shape[time_axis]) * 100
            )

            assert max_freq_mask_perc <= max_signal_override_perc, (
                f"augmentation parameters exceed {max_signal_override_perc}% of the signal HEIGHT: max_freq_mask_perc={max_freq_mask_perc}%"
            )
            assert max_time_mask_perc <= max_signal_override_perc, (
                f"augmentation parameters exceed {max_signal_override_perc}% of the signal WIDTH: max_time_mask_perc={max_time_mask_perc}%"
            )

    def _augment(self, sample: np.ndarray) -> np.ndarray:
        sample_augmented = sample.copy()
        freq_masks = self.cfg.freq_masks
        time_masks = self.cfg.time_masks

        freq_domain = self.shape[1] if sample.ndim == 3 else self.shape[0]
        time_domain = self.shape[2] if sample.ndim == 3 else self.shape[1]
        for _ in range(freq_masks):
            freq_mask_width = np.random.randint(1, self.cfg.freq_mask_width + 1)
            pos = np.random.randint(0, (freq_domain - freq_mask_width + 1))
            if sample.ndim == 3:
                sample_augmented[:, pos : (pos + freq_mask_width), :].fill(0.0)
            else:
                sample_augmented[pos : (pos + freq_mask_width), :].fill(0.0)

        for _ in range(time_masks):
            time_mask_width = np.random.randint(1, self.cfg.time_mask_width + 1)
            pos = np.random.randint(0, (time_domain - time_mask_width) + 1)
            if sample.ndim == 3:
                sample_augmented[:, :, pos : (pos + time_mask_width)].fill(0.0)
            else:
                sample_augmented[:, pos : (pos + time_mask_width)].fill(0.0)

        return sample_augmented

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        if self.dataset_type == DatasetType.TRAIN and self.cfg.augment:
            sample = self._augment(self.samples[idx])
        else:
            sample = self.samples[idx]

        return (torch.from_numpy(sample), self.labels[idx])


OPTIMIZER_MAP: dict[str, type[optim.Optimizer]] = {
    "SGD": optim.SGD,
    "ADAM": optim.Adam,
    "ADAMW": optim.AdamW,
}

SCHEDULER_MAP: dict[str, type[optim.lr_scheduler.LRScheduler]] = {
    "PLATEAU": optim.lr_scheduler.ReduceLROnPlateau,
    "COSINE": optim.lr_scheduler.CosineAnnealingLR,
    "STEP": optim.lr_scheduler.StepLR,
}


MODEL_CONFIG_MAP: dict[ModelType, type[ConfigCNN] | type[ConfigLSTM]] = {
    ModelType.CNN: ConfigCNN,
    ModelType.LSTM: ConfigLSTM,
}

CNN_LAYER_MAP: dict[CNNLayers, type[LayerConv] | type[LayerPool]] = {
    CNNLayers.CONV: LayerConv,
    CNNLayers.POOL: LayerPool,
}


@dataclass(frozen=True)
class ArgsCLI:
    cfg_path: Path
    csv_path: Path
    save_model: bool = False
    cross_val_csv_path: Path | None = None
    model_path: Path | None = None
    eval_folds: list[int] | None = None


class CMInfo(TypedDict):
    preds: np.ndarray
    labels: np.ndarray
    class_names: list[str]


class TrainRunInfo(TypedDict):
    net: nn.Module
    cfg_instance: ConfigCNN | ConfigLSTM
    content: dict
    run_id: str
    args: ArgsCLI
    train_info_for_plots: dict
    emit: Callable[[str, str, str, dict], None]
    cv_run_id: str | None
    cm_info: CMInfo


class CLIArgumentError(Exception):
    exit_code = 3
