from dataclasses import fields
from pathlib import Path

from infra.data_models import (
    MODEL_CONFIG_MAP,
    ModelType,
    ReprType,
)


def normalize_config(cfg: dict, path: Path) -> dict:
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

    for f in fields(cfg_class):
        key = f.name
        k_type = f.type

        if key not in cfg:
            raise KeyError(f"missing key {path.stem}.[model/run].{key}")
        val = cfg[key]
        if k_type not in (list[int], float | None):
            assert isinstance(val, k_type), (
                f"invalid key type {path.stem}.[model/run].{key}: '{type(val)}', expected '{k_type}'"
            )

    return cfg
