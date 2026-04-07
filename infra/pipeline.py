import logging
from time import perf_counter

from infra.cli_utils import normalize_and_validate_config, set_seed, validate_eval_folds
from infra.data_models import ArgsCLI
from infra.io_utils import (
    load_yaml_config,
    save_cross_val_artifacts,
    save_eval_artifacts,
    save_train_run_info,
)
from infra.log_utils import make_emit
from models.cross_val import cross_validation_loop
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
            "cross_val_csv_path": str(args.cross_val_csv_path),
            "csv_path": str(args.csv_path),
            "cfg_path": str(args.cfg_path),
            "save_model": args.save_model,
        },
    )

    cfg_dict_raw = load_yaml_config(args.cfg_path)
    cfg_dict_norm = normalize_and_validate_config(cfg_dict_raw, args.cfg_path)

    seed = cfg_dict_norm["seed"]
    set_seed(seed)

    if pipe_type == "train":
        if args.cross_val_csv_path is not None:
            content, train_runs, cv_train_info = cross_validation_loop(
                cfg_dict_norm, logger=logger, emit=emit, cv_run_id=run_id, args=args
            )

            for train_run in train_runs:
                save_train_run_info(**train_run)

            save_cross_val_artifacts(
                content, run_id, args=args, cv_train_info=cv_train_info, emit=emit
            )
        else:
            net, cfg, content, train_info_for_plots = training_loop(
                cfg_dict_norm, emit=emit, run_id=run_id, args=args
            )
            save_train_run_info(
                net,
                cfg,
                content,
                run_id,
                args=args,
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
        payload={
            "type": pipe_type,
            "cross_val_csv_path": str(args.cross_val_csv_path),
            "elapsed_time": round((t2 - t1), 4),
        },
    )
