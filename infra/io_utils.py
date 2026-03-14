from pathlib import Path

import numpy as np
import yaml


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
