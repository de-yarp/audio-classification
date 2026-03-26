import random
from dataclasses import fields
from pathlib import Path
from typing import get_type_hints

import numpy as np
import torch

from infra.data_models import (
    CNN_LAYER_MAP,
    MODEL_CONFIG_MAP,
    ArgsCLI,
    CLIArgumentError,
    CNNLayers,
    LayerConv,
    LayerPool,
    ModelType,
    OptimizerType,
    PoolType,
    ReprType,
    SchedulerType,
)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def _validate_cnn_layers(val: dict, key: str, path: Path):
    new_layers = []
    for l_idx, layer in enumerate(val, 0):
        l_type_str = layer["type"].lower()
        try:
            l_type = CNNLayers(l_type_str)
            layer_class = CNN_LAYER_MAP[l_type]
        except ValueError as e:
            msg = f"invalid model.layers.[{l_idx}] '{l_type_str}' in config {path}, expected {[e.value for e in CNNLayers]}"
            raise ValueError(msg) from e

        new_layer = {}
        for f1 in fields(layer_class):
            val_1 = layer[f1.name]
            assert isinstance(val_1, f1.type), (
                f"invalid key type {path.stem}.[model/run].{key}.{f1.name}: '{type(val_1)}', expected '{f1.type}'"
            )
            new_layer[f1.name] = val_1

        new_layers.append(layer_class(**new_layer))
    return new_layers


def _validate_cfg_folds(cfg: dict, path: Path) -> None:
    val_folds = cfg["folds_val"]
    train_folds = cfg["folds_train"]

    val_folds_set = set(val_folds)
    train_folds_set = set(train_folds)

    assert len(val_folds) == len(val_folds_set), (
        f"model.folds_val '{val_folds}' contains duplicates in config {path}"
    )
    assert len(train_folds) == len(train_folds_set), (
        f"model.folds_train '{train_folds}' contains duplicates in config {path}"
    )
    intersection = val_folds_set & train_folds_set
    assert not intersection, (
        f"model.folds_val and model.folds_train contain common values {intersection} in config {path}"
    )


def normalize_and_validate_config(cfg: dict, path: Path) -> dict:
    model_type_str = cfg["model_type"].strip().lower()
    repr_type_str = cfg["repr_type"].strip().lower()
    optimizer_str = cfg["optimizer"].strip().upper()
    if cfg["scheduler"] is not None:
        scheduler_str = cfg["scheduler"].strip().upper()
    else:
        scheduler_str = None

    try:
        model_type = ModelType(model_type_str)
    except ValueError as e:
        msg = f"invalid model.type '{model_type_str}' in config {path}, expected {[e.value for e in ModelType]}"
        raise ValueError(msg) from e
    try:
        repr_type = ReprType(repr_type_str)
    except ValueError as e:
        msg = f"invalid model.repr_type '{repr_type_str}' in config {path}, expected {[e.value for e in ReprType]}"
        raise ValueError(msg) from e
    try:
        optimizer_type = OptimizerType(optimizer_str)
    except ValueError as e:
        msg = f"invalid run.optimizer '{optimizer_str}' in config {path}, expected {[e.value for e in OptimizerType]}"
        raise ValueError(msg) from e

    if scheduler_str is not None:
        try:
            scheduler_type = SchedulerType(scheduler_str)

        except ValueError as e:
            msg = f"invalid run.scheduler '{scheduler_str}' in config {path}, expected {[e.value for e in SchedulerType]}"
            raise ValueError(msg) from e
    else:
        scheduler_type = None

    if model_type == ModelType.CNN:
        pool_type_str = cfg["pool_type"].strip().lower()
        try:
            pool_type = PoolType(pool_type_str)
        except ValueError as e:
            msg = f"invalid model.pool_type '{pool_type_str}' in config {path}, expected {[e.value for e in PoolType]}"
            raise ValueError(msg) from e
        cfg["pool_type"] = pool_type

    cfg["model_type"] = model_type
    cfg["repr_type"] = repr_type
    cfg["optimizer"] = optimizer_type
    cfg["scheduler"] = scheduler_type

    cfg_class = MODEL_CONFIG_MAP[model_type]

    cfg_class_types = get_type_hints(cfg_class)

    for f in fields(cfg_class):
        key = f.name
        k_type = cfg_class_types[key]

        if key not in cfg:
            raise KeyError(f"missing key {path.stem}.[model/run].{key}")
        val = cfg[key]

        if k_type == list[LayerConv | LayerPool]:
            cfg[key] = _validate_cnn_layers(val, key, path)

        elif k_type == list[int]:
            for i in val:
                assert isinstance(i, int), (
                    f"invalid key type {path.stem}.[model/run].{key}: '{type(i)}', expected 'int'"
                )

        elif k_type == float | None:
            assert isinstance(val, float) or val is None, (
                f"invalid key type {path.stem}.[model/run].{key}: '{type(val)}', expected '{k_type}'"
            )

        elif k_type == int | None:
            assert isinstance(val, int) or val is None, (
                f"invalid key type {path.stem}.[model/run].{key}: '{type(val)}', expected '{k_type}'"
            )

        elif k_type == SchedulerType | None:
            assert isinstance(val, SchedulerType) or val is None, (
                f"invalid key type {path.stem}.[model/run].{key}: '{type(val)}', expected '{k_type}'"
            )

        else:
            assert isinstance(val, k_type), (
                f"invalid key type {path.stem}.[model/run].{key}: '{type(val)}', expected '{k_type}'"
            )

    _validate_cfg_folds(cfg, path)

    return cfg


def validate_args_paths(
    cfg_path: Path,
    csv_path: Path,
    model_path: Path | None = None,
    cross_val_csv_path: Path | None = None,
) -> None:
    try:
        if cfg_path.stat().st_size == 0:
            msg = f"cfg_path: {cfg_path} is empty"
            raise CLIArgumentError(msg)
    except FileNotFoundError:
        msg = f"cfg_path: {cfg_path} not found"
        raise CLIArgumentError(msg)

    if cfg_path.suffix != ".yaml":
        msg = f"cfg_path: invalid suffix '{cfg_path.suffix}', expected '.yaml'"
        raise CLIArgumentError(msg)

    try:
        csv_path.stat().st_size
    except FileNotFoundError:
        msg = f"csv_path: {csv_path} not found"
        raise CLIArgumentError(msg)

    if csv_path.suffix != ".csv":
        msg = f"csv_path: invalid suffix '{csv_path.suffix}', expected '.csv'"
        raise CLIArgumentError(msg)

    if cross_val_csv_path is not None:
        try:
            cross_val_csv_path.stat().st_size
        except FileNotFoundError:
            msg = f"cross_val_csv_path: {cross_val_csv_path} not found"
            raise CLIArgumentError(msg)

        if cross_val_csv_path.suffix != ".csv":
            msg = f"cross_val_csv_path: invalid suffix '{cross_val_csv_path.suffix}', expected '.csv'"
            raise CLIArgumentError(msg)

    if model_path is None:
        return

    try:
        if model_path.stat().st_size == 0:
            msg = f"model_path: {str(model_path)} is empty"
            raise CLIArgumentError(msg)
    except FileNotFoundError:
        msg = f"model_path: {str(model_path)} not found"
        raise CLIArgumentError(msg)

    if model_path.suffix != ".pt":
        msg = f"model_path: invalid suffix '{model_path.suffix}', expected '.pt'"
        raise CLIArgumentError(msg)

    if model_path.stem != cfg_path.stem:
        msg = f"model_path/cfg_path: stem mismatch ['{str(model_path.stem)}' vs '{str(cfg_path.stem)}']; checkpoint and config must be of the same run"
        raise CLIArgumentError(msg)


def validate_eval_folds(args: ArgsCLI, cfg: dict) -> None:
    input_folds = args.eval_folds
    input_folds_set = set(input_folds)
    if len(input_folds) != len(input_folds_set):
        msg = "eval_folds: contains duplicate values"
        raise CLIArgumentError(msg)

    cfg_folds = set(cfg["folds_val"] + cfg["folds_train"])
    intersection = cfg_folds & input_folds_set

    if intersection:
        msg = f"eval_folds: contains common values with config set folds, intersection={intersection}"
        raise CLIArgumentError(msg)
