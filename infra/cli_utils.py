from dataclasses import fields
from pathlib import Path
from typing import get_type_hints

from infra.data_models import (
    CNN_LAYER_MAP,
    MODEL_CONFIG_MAP,
    ArgsCLI,
    CLIArgumentError,
    CNNLayers,
    LayerConv,
    LayerPool,
    ModelType,
    ReprType,
)


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


def normalize_and_validate_config(cfg: dict, path: Path) -> dict:
    model_type_str = cfg["model_type"].strip().lower()
    repr_type_str = cfg["repr_type"].strip().lower()
    optimizer_str = cfg["optimizer"].strip().upper()

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

    cfg["model_type"] = model_type
    cfg["repr_type"] = repr_type
    cfg["optimizer"] = optimizer_str

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
                assert isinstance(i, int)

        elif k_type == float | None:
            assert isinstance(val, float) or val is None, (
                f"invalid key type {path.stem}.[model/run].{key}: '{type(val)}', expected '{k_type}'"
            )

        else:
            assert isinstance(val, k_type), (
                f"invalid key type {path.stem}.[model/run].{key}: '{type(val)}', expected '{k_type}'"
            )

    return cfg


def validate_paths(cfg_path: Path, csv_path: Path) -> None:
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


def validate_cli_args(args: ArgsCLI) -> None:
    validate_paths(args.cfg_path, args.csv_path)
