import logging
import random
from time import perf_counter

import numpy as np
import torch

from infra.cli_utils import normalize_and_validate_config
from infra.data_models import ArgsCLI
from infra.io_utils import load_yaml_config, save_run_info
from infra.log_utils import make_emit
from models.train import training_loop

COMPONENT = __name__


def pipe_run(args: ArgsCLI, *, logger: logging.Logger, run_id: str) -> None:
    t1 = perf_counter()

    emit = make_emit(logger, run_id)

    emit(
        level="INFO",
        component=COMPONENT,
        event="start_pipeline",
        payload={
            "csv_path": str(args.csv_path),
            "cfg_path": str(args.cfg_path),
            "save_model": args.save_model,
        },
    )

    cfg_dict_raw = load_yaml_config(args.cfg_path)
    cfg_dict_norm = normalize_and_validate_config(cfg_dict_raw, args.cfg_path)

    seed = cfg_dict_norm["seed"]
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    net, cfg, content = training_loop(
        cfg_dict_norm, logger=logger, run_id=run_id, args=args
    )
    save_run_info(
        args.save_model, net, cfg, args.cfg_path, content, args.csv_path, emit=emit
    )

    t2 = perf_counter()

    emit(
        level="INFO",
        component=COMPONENT,
        event="end_pipeline",
        payload={"elapsed_time": round((t2 - t1), 4)},
    )
