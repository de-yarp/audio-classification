from pathlib import Path

import librosa as lbr
import numpy as np

from infra.data_models import FeatureConfig

from .data_models import ESC_50_PROCESSED_PATH, ESC_50_RAW_PATH
from .io_utils import load_feature_config, save_features_npy


def get_features_esc50(
    in_dir: Path = ESC_50_RAW_PATH / "audio",
    out_dir: Path = ESC_50_PROCESSED_PATH,
    cfg_path: Path = Path("config") / "features.yaml",
) -> None:
    cfg = load_feature_config(cfg_path)
    for path in in_dir.glob("*.wav"):
        audio, sr = lbr.load(path)
        mel_spec, mfcc = _compute_features_esc50(audio, sr, cfg)
        save_features_npy(mel_spec, mfcc, out_dir, path.stem)


def _compute_features_esc50(
    audio: np.ndarray, sr: int | float, cfg: FeatureConfig
) -> tuple[np.ndarray, np.ndarray]:
    mel_spec = lbr.feature.melspectrogram(
        y=audio, sr=sr, n_fft=cfg.n_fft, hop_length=cfg.hop_length, n_mels=cfg.n_mels
    )
    mel_spec = lbr.power_to_db(mel_spec)

    mfcc = lbr.feature.mfcc(S=mel_spec, n_mfcc=cfg.n_mfcc)
    if cfg.include_deltas:
        mfcc_delta = lbr.feature.delta(mfcc)
        mfcc_delta2 = lbr.feature.delta(mfcc_delta)
        mfcc = np.vstack((mfcc, mfcc_delta, mfcc_delta2))

    return mel_spec, mfcc
