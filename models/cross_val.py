import logging
import uuid
from itertools import combinations
from typing import Callable

from infra.cli_utils import set_seed
from infra.data_models import ArgsCLI, TrainRunInfo
from infra.log_utils import make_emit, now_ts_iso

from .train import training_loop

COMPONENT = __name__


def _get_fold_combinations(cfg_dict: dict) -> list[tuple[list[int], list[int]]]:
    fold_combinations: list[tuple[list[int], list[int]]] = []
    folds_train = cfg_dict["folds_train"]
    folds_val = cfg_dict["folds_val"]
    available_folds = folds_train + folds_val
    folds_val_combinations = combinations(available_folds, len(folds_val))
    for fv_tuple in folds_val_combinations:
        fv = list(fv_tuple)
        ft = [f for f in available_folds if f not in fv]
        fold_combinations.append((ft, fv))

    return fold_combinations


def cross_validation_loop(
    cfg_dict: dict,
    *,
    logger: logging.Logger,
    emit: Callable[[str, str, str, dict], None],
    cv_run_id: str,
    args: ArgsCLI,
) -> tuple[dict, list[TrainRunInfo], list[dict]]:

    emit(
        level="INFO",
        component=COMPONENT,
        event="start_cross_validation",
    )
    seed = cfg_dict["seed"]

    fold_combinations = _get_fold_combinations(cfg_dict)
    cv_train_info: list[dict] = []
    train_runs: list[TrainRunInfo] = []
    child_run_ids = []

    cfg_dict_child = cfg_dict.copy()
    for folds_train, folds_val in fold_combinations:
        set_seed(seed)
        run_id_child = str(uuid.uuid4())
        emit_child = make_emit(logger, run_id_child)
        child_run_ids.append(run_id_child)

        cfg_dict_child["folds_train"] = folds_train
        cfg_dict_child["folds_val"] = folds_val

        net, cfg, content, train_info_for_plots = training_loop(
            cfg_dict_child,
            emit=emit_child,
            run_id=run_id_child,
            args=args,
            cv_run_id=cv_run_id,
        )

        save_params: TrainRunInfo = {
            "net": net,
            "cfg_instance": cfg,
            "content": content,
            "run_id": run_id_child,
            "args": args,
            "cv_run_id": cv_run_id,
            "train_info_for_plots": train_info_for_plots,
            "emit": emit_child,
        }
        train_runs.append(save_params)

        cv_train_info.append(train_info_for_plots)

        # save_train_run_info(
        #     net,
        #     cfg,
        #     content,
        #     run_id_child,
        #     args=args,
        #     cv_run_id=cv_run_id,
        #     train_info_for_plots=train_info_for_plots,
        #     emit=emit_child,
        # )

    emit(
        level="INFO",
        component=COMPONENT,
        event="finish_cross_validation",
        payload={"child_run_ids": child_run_ids},
    )

    content = {
        "ts": now_ts_iso(),
        "cv_run_id": cv_run_id,
        "child_run_ids": ";".join(child_run_ids),
    }

    return content, train_runs, cv_train_info
