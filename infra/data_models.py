from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class FeatureConfig:
    n_fft: int
    hop_length: int
    n_mels: int
    n_mfcc: int
    include_deltas: bool


class ReprType(Enum):
    MEL = "mel"
    MFCC = "mfcc"


ESC_50_RAW_PATH = Path("data") / "raw" / "esc50"
ESC_50_PROCESSED_PATH = Path("data") / "processed" / "esc50"


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
