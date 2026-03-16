import logging
import random
from time import perf_counter

import numpy as np
import torch

from infra.cli_utils import normalize_and_validate_config, validate_eval_folds
from infra.data_models import ArgsCLI
from infra.io_utils import load_yaml_config, save_eval_artifacts, save_train_run_info
from infra.log_utils import make_emit
from models.eval import evaluate_model
from models.train import training_loop

COMPONENT = __name__


def pipe_run(args: ArgsCLI, *, logger: logging.Logger, run_id: str) -> None:
    if args.model_path is None:
        pipe_type = "train"
    else:
        pipe_type = "eval"

    t1 = perf_counter()

    emit = make_emit(logger, run_id)

    emit(
        level="INFO",
        component=COMPONENT,
        event="start_pipeline",
        payload={
            "type": pipe_type,
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

    if pipe_type == "train":
        net, cfg, content, train_info_for_plots = training_loop(
            cfg_dict_norm, emit=emit, run_id=run_id, args=args
        )
        save_train_run_info(
            args.save_model,
            net,
            cfg,
            args.cfg_path,
            args.csv_path,
            content,
            run_id,
            train_info_for_plots=train_info_for_plots,
            emit=emit,
        )
    elif pipe_type == "eval":
        validate_eval_folds(args, cfg_dict_norm)
        preds, labels, class_names, content = evaluate_model(
            cfg_dict_norm, emit=emit, run_id=run_id, args=args
        )
        save_eval_artifacts(
            preds, labels, content, run_id, class_names, emit=emit, args=args
        )

    t2 = perf_counter()

    emit(
        level="INFO",
        component=COMPONENT,
        event="end_pipeline",
        payload={"type": pipe_type, "elapsed_time": round((t2 - t1), 4)},
    )
