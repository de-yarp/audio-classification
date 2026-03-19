import traceback
import uuid
from pathlib import Path
from typing import Annotated

import typer

from .cli_utils import validate_args_paths
from .data_models import ArgsCLI, CLIArgumentError
from .log_utils import make_emit, setup_logger
from .pipeline import pipe_run

COMPONENT = __name__

app = typer.Typer(rich_markup_mode="markdown")

logger = setup_logger()


@app.command()
def eval(
    cfg_path: Annotated[Path, typer.Argument(..., help="path to your model config")],
    csv_path: Annotated[
        Path, typer.Argument(..., help="path to your experiments tracker .csv")
    ],
    model_path: Annotated[
        Path, typer.Argument(..., help="path to your model checkpoint")
    ],
    eval_folds: Annotated[list[int], typer.Option(..., help="number of eval folds")],
):
    run_id = str(uuid.uuid4())
    emit = make_emit(logger, run_id)
    try:
        args = ArgsCLI(
            cfg_path=cfg_path,
            csv_path=csv_path,
            model_path=model_path,
            eval_folds=eval_folds,
        )
        validate_args_paths(args.cfg_path, args.csv_path, args.model_path)
        pipe_run(args, logger=logger, run_id=run_id)
    except CLIArgumentError as e:
        emit(
            level="ERROR",
            component=COMPONENT,
            event="invalid_cli_argument",
            payload={"error_msg": str(e)},
        )
        raise
    except Exception as e:
        emit(
            level="ERROR",
            component=COMPONENT,
            event="unexpected_error",
            payload={"error_msg": str(e), "traceback": traceback.format_exc()},
        )
        raise


@app.command()
def train(
    cfg_path: Annotated[Path, typer.Argument(..., help="path to your model config")],
    csv_path: Annotated[
        Path, typer.Argument(..., help="path to your experiments tracker .csv")
    ],
    cross_val_csv_path: Annotated[
        Path | None,
        typer.Option("--cross-val", help="path to cross validation .csv tracker"),
    ] = None,
    save_model: Annotated[
        bool,
        typer.Option(
            "--save-model", help="include, if you want to save a model checkpoint"
        ),
    ] = False,
):
    run_id = str(uuid.uuid4())
    emit = make_emit(logger, run_id)
    try:
        args = ArgsCLI(
            cfg_path=cfg_path,
            csv_path=csv_path,
            save_model=save_model,
            cross_val_csv_path=cross_val_csv_path,
        )
        validate_args_paths(
            args.cfg_path, args.csv_path, args.model_path, args.cross_val_csv_path
        )
        pipe_run(args, logger=logger, run_id=run_id)
    except CLIArgumentError as e:
        emit(
            level="ERROR",
            component=COMPONENT,
            event="invalid_cli_argument",
            payload={"error_msg": str(e)},
        )
        raise
    except Exception as e:
        emit(
            level="ERROR",
            component=COMPONENT,
            event="unexpected_error",
            payload={"error_msg": str(e), "traceback": traceback.format_exc()},
        )
        raise
