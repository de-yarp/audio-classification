from pathlib import Path

import numpy as np
import pytest

from infra.data_models import FeatureConfig
from infra.preprocessing import get_features_esc50


@pytest.mark.slow
class TestPreprocessing:
    @pytest.fixture(autouse=True, scope="class")
    def load_data(self, request, tmp_path_factory) -> None:
        request.cls.tmp_root = tmp_path_factory.mktemp("data")
        request.cls.mel_dir = request.cls.tmp_root / "mel"
        request.cls.mfcc_dir = request.cls.tmp_root / "mfcc"

        request.cls.tmp_root_2 = tmp_path_factory.mktemp("data")
        request.cls.mel_dir_2 = request.cls.tmp_root_2 / "mel"
        request.cls.mfcc_dir_2 = request.cls.tmp_root_2 / "mfcc"

        request.cls.cfg_path = Path("config") / "features.yaml"
        get_features_esc50(out_dir=request.cls.tmp_root, cfg_path=request.cls.cfg_path)

    def test_shape(self) -> None:
        cfg = FeatureConfig.from_yaml(self.cfg_path)
        mel_spec = np.load(next(self.mel_dir.glob("*.npy")))
        assert mel_spec.shape[0] == cfg.n_mels
        mfcc_spec = np.load(next(self.mfcc_dir.glob("*.npy")))
        if cfg.include_deltas:
            assert mfcc_spec.shape[0] == cfg.n_mfcc * 3
        else:
            assert mfcc_spec.shape[0] == cfg.n_mfcc

    def test_idempotency(self) -> None:
        with pytest.raises(FileExistsError):
            get_features_esc50(out_dir=self.tmp_root, cfg_path=self.cfg_path)

    def test_output_across_runs(self) -> None:
        get_features_esc50(out_dir=self.tmp_root_2, cfg_path=self.cfg_path)
        mel_spec_1 = np.load(next(self.mel_dir.glob("*.npy")))
        mel_spec_2 = np.load(next(self.mel_dir_2.glob("*.npy")))
        assert np.array_equal(mel_spec_1, mel_spec_2)

        mfcc_spec_1 = np.load(next(self.mfcc_dir.glob("*.npy")))
        mfcc_spec_2 = np.load(next(self.mfcc_dir_2.glob("*.npy")))
        assert np.array_equal(mfcc_spec_1, mfcc_spec_2)
