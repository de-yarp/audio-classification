import traceback
import uuid
from pathlib import Path
from typing import Annotated

import typer

from .cli_utils import validate_cli_args
from .data_models import ArgsCLI, CLIArgumentError
from .log_utils import make_emit, setup_logger
from .pipeline import pipe_run

COMPONENT = __name__

app = typer.Typer(rich_markup_mode="markdown")

logger = setup_logger()


@app.command()
def main(
    cfg_path: Annotated[Path, typer.Argument(..., help="path to your model config")],
    csv_path: Annotated[
        Path, typer.Argument(..., help="path to your experiments tracker .csv")
    ],
    save_model: Annotated[
        bool,
        typer.Option(
            "--save_model", help="include, if you want to save a model checkpoint"
        ),
    ] = False,
):
    run_id = str(uuid.uuid4())
    emit = make_emit(logger, run_id)
    try:
        args = ArgsCLI(cfg_path, csv_path, save_model)
        validate_cli_args(args)
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
