from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from infra.data_models import ESC_50_RAW_PATH, AudioDataset, ReprType
from infra.preprocessing import get_features_esc50


@pytest.mark.slow
class TestAudioDataset:
    @pytest.fixture(autouse=True, scope="class")
    def load_data(self, request, tmp_path_factory) -> None:
        request.cls.tmp_root = tmp_path_factory.mktemp("data")
        request.cls.cfg_path = Path("config") / "features.yaml"
        request.cls.csv_path = ESC_50_RAW_PATH / "meta" / "esc50.csv"

        repr_type = ReprType.MEL
        request.cls.repr_dir = request.cls.tmp_root / repr_type.value

        get_features_esc50(out_dir=request.cls.tmp_root, cfg_path=request.cls.cfg_path)

        request.cls.folds = [1]
        request.cls.dataset = AudioDataset(
            repr_type,
            request.cls.folds,
            root_dir=request.cls.tmp_root,
            csv_file=request.cls.csv_path,
        )

    def test_len(self) -> None:
        assert len(self.dataset) == len(self.folds) * 400

    def test_getitem(self) -> None:
        meta_csv = pd.read_csv(self.csv_path, delimiter=",")
        idx = 0
        sample, label = self.dataset[idx]

        assert isinstance(sample, torch.Tensor)
        assert isinstance(label, int)

        if self.folds == [1]:
            sample_path = self.repr_dir / Path(
                meta_csv["filename"].iloc[idx]
            ).with_suffix(".npy")
            label_real = meta_csv["target"].iloc[idx]
            sample_real = np.load(sample_path)

            assert np.array_equal(sample.numpy(), sample_real)
            assert label == label_real
