from dataclasses import asdict, dataclass, fields
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import yaml


@dataclass(frozen=True)
class FeatureConfig:
    n_fft: int
    hop_length: int
    n_mels: int
    n_mfcc: int
    include_deltas: bool

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


ESC_50_RAW_PATH = Path("data") / "raw" / "esc50"
ESC_50_PROCESSED_PATH = Path("data") / "processed" / "esc50"
LOGS_DIR_PATH = Path("logs")
LOG_NAME = "log.jsonl"
MODEL_CHECKPOINTS_DIR_PATH = Path("runs") / "checkpoints"
MODEL_CONFIGS_DIR_PATH = Path("runs") / "configs"


class AudioDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        repr_type: ReprType,
        folds: list[int],
        *,
        root_dir: Path = ESC_50_PROCESSED_PATH,
        csv_file: Path = ESC_50_RAW_PATH / "meta" / "esc50.csv",
    ):
        self.folds = folds
        self.repr_dir = root_dir / repr_type.value
        self.meta_df = pd.read_csv(csv_file, delimiter=",")

        self.samples, self.labels = self._load_samples()

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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return (torch.from_numpy(self.samples[idx]), self.labels[idx])


@dataclass(frozen=True)
class LayerConv:
    kernel_count: int
    kernel_size: int
    stride: int
    padding: int
    batch_norm: bool = False
    type: str = "conv"


@dataclass(frozen=True)
class LayerPool:
    kernel_size: int
    stride: int
    padding: int
    type: str = "conv"


@dataclass(frozen=True)
class ConfigCNN:
    # model config
    model_type: ModelType
    repr_type: ReprType
    conv_layers: list[LayerConv | LayerPool]
    fc_layers: list[int]
    num_classes: int

    # run config
    seed: int
    batch_size: int
    folds_train: list[int]
    folds_val: list[int]
    num_epochs: int
    optimizer: str
    lr: float
    momentum: float | None

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
    """TODO: finish the config class for arguments lstm needs"""

    # model config
    model_type: ModelType
    repr_type: ReprType
    num_classes: int

    # run config
    seed: int
    batch_size: int
    folds_train: list[int]
    folds_val: list[int]
    num_epochs: int
    optimizer: str
    lr: float
    momentum: float | None
    ...

    @classmethod
    def from_dict(cls, input: dict):
        inner_keys = {f.name for f in fields(cls)}
        data = {k: v for k, v in input.items() if k in inner_keys}

        return cls(**data)

    def to_dict(self) -> dict: ...


OPTIMIZER_MAP: dict[str, type[optim.Optimizer]] = {
    "SGD": optim.SGD,
    "ADAM": optim.Adam,
    "ADAMW": optim.AdamW,
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
    save_model: bool


class CLIArgumentError(Exception):
    exit_code = 3
