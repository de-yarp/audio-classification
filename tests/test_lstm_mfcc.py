from pathlib import Path

import pytest
import torch

from infra.cli_utils import normalize_and_validate_config
from infra.data_models import (
    ConfigLSTM,
    ModelType,
    OptimizerType,
    ReprType,
    SchedulerType,
)
from models.lstm_mfcc import MFCC_LSTM

_BASE_CFG = dict(
    model_type=ModelType.LSTM,
    repr_type=ReprType.MFCC,
    mfcc_deltas=True,
    hidden_size=32,
    num_layers=1,
    dropout=0.0,
    fc_layers=[64],
    num_classes=50,
    seed=0,
    batch_size=2,
    folds_train=[1],
    folds_val=[2],
    num_epochs=1,
    optimizer=OptimizerType.ADAM,
    lr=1e-3,
    momentum=None,
    weight_decay=0.0,
    warmup_lr=False,
    warmup_epochs=None,
    warmup_lr_val=None,
    scheduler=None,
    factor=None,
    patience=None,
    min_lr=None,
    step_size=None,
    augment=False,
    freq_masks=None,
    freq_mask_width=None,
    time_masks=None,
    time_mask_width=None,
)

_BATCH = 2
_TIME = 216
_INPUT = torch.zeros(_BATCH, 120, _TIME)  # (batch, input_size, time)


def _make_cfg(**overrides) -> ConfigLSTM:
    return ConfigLSTM(**{**_BASE_CFG, **overrides})


class TestMFCC_LSTM:
    def test_default_output_shape(self):
        cfg = _make_cfg()
        net = MFCC_LSTM(cfg=cfg)
        out = net(_INPUT)
        assert out.shape == (_BATCH, cfg.num_classes)

    def test_bidirectional_mean_output_shape(self):
        cfg = _make_cfg(bidirectional=True, pooling="mean")
        net = MFCC_LSTM(cfg=cfg)
        out = net(_INPUT)
        assert out.shape == (_BATCH, cfg.num_classes)

    def test_bidirectional_last_output_shape(self):
        cfg = _make_cfg(bidirectional=True, pooling="last")
        net = MFCC_LSTM(cfg=cfg)
        out = net(_INPUT)
        assert out.shape == (_BATCH, cfg.num_classes)

    def test_max_pooling_output_shape(self):
        cfg = _make_cfg(pooling="max")
        net = MFCC_LSTM(cfg=cfg)
        out = net(_INPUT)
        assert out.shape == (_BATCH, cfg.num_classes)


class TestLSTMConfigValidation:
    def _raw_cfg(self, **overrides) -> dict:
        raw = dict(
            model_type="lstm",
            repr_type="mfcc",
            mfcc_deltas=True,
            hidden_size=32,
            num_layers=1,
            dropout=0.0,
            fc_layers=[64],
            num_classes=50,
            seed=0,
            batch_size=2,
            folds_train=[1],
            folds_val=[2],
            num_epochs=1,
            optimizer="ADAM",
            lr=1e-3,
            momentum=None,
            weight_decay=0.0,
            warmup_lr=False,
            warmup_epochs=None,
            warmup_lr_val=None,
            scheduler=None,
            factor=None,
            patience=None,
            min_lr=None,
            step_size=None,
            augment=False,
            freq_masks=None,
            freq_mask_width=None,
            time_masks=None,
            time_mask_width=None,
        )
        raw.update(overrides)
        return raw

    def test_invalid_pooling_raises_value_error(self):
        raw = self._raw_cfg(pooling="attention")
        with pytest.raises(ValueError, match="pooling"):
            normalize_and_validate_config(raw, Path("dummy.yaml"))

    def test_missing_bidirectional_uses_default(self):
        raw = self._raw_cfg()
        result = normalize_and_validate_config(raw, Path("dummy.yaml"))
        assert result["bidirectional"] is False

    def test_missing_pooling_uses_default(self):
        raw = self._raw_cfg()
        result = normalize_and_validate_config(raw, Path("dummy.yaml"))
        assert result["pooling"] == "last"
