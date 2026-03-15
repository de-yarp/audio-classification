import datetime as dt
import json
import logging
import logging.handlers
import sys
from typing import Any, Callable, override

from .data_models import LOG_NAME, LOGS_DIR_PATH


def now_ts_iso() -> str:
    return (
        dt.datetime.now(tz=dt.timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


class JSONFormatter(logging.Formatter):
    @override
    def format(self, rec: logging.LogRecord) -> str:
        obj = {
            "ts": now_ts_iso(),
            "level": rec.levelname,
            "event": rec.getMessage(),
        }

        run_id = getattr(rec, "run_id", None)
        component = getattr(rec, "component", None)
        payload = getattr(rec, "payload", None)
        if run_id:
            obj["run_id"] = run_id
        if component:
            obj["component"] = component
        if payload and isinstance(payload, dict):
            obj.update(payload)

        return json.dumps(obj)


def setup_logger(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(level)

    h_file = logging.handlers.RotatingFileHandler(
        LOGS_DIR_PATH / LOG_NAME, encoding="utf-8", maxBytes=int(1e7), backupCount=3
    )
    h_file.setLevel(logging.INFO)
    h_err = logging.StreamHandler(sys.stderr)
    h_err.setLevel(logging.WARNING)

    f = JSONFormatter()
    h_file.setFormatter(f)
    h_err.setFormatter(f)

    logger.addHandler(h_file)
    logger.addHandler(h_err)

    return logger


def make_emit(
    logger: logging.Logger, run_id: str
) -> Callable[[str, str, str, dict], None]:
    LEVELS_MAP = {"INFO": logging.INFO, "WARN": logging.WARNING, "ERROR": logging.ERROR}

    def emit(level: str, component: str, event: str, payload: dict = {}) -> None:
        logger.log(
            level=LEVELS_MAP.get(level.upper(), logging.INFO),
            msg=event,
            extra={"run_id": run_id, "component": component, "payload": payload},
        )

    return emit


def parse_run_logs(run_id: str) -> list[dict[str, Any]]:
    run_logs: list[dict[str, Any]] = []
    logs_paths = LOGS_DIR_PATH.glob(f"{LOG_NAME}*")
    for log in logs_paths:
        with log.open(mode="r", encoding="utf-8") as f:
            for line in f:
                json_line = json.loads(line.strip())
                if "run_id" not in json_line:
                    continue
                if json_line["run_id"] == run_id:
                    run_logs.append(json_line)

    return run_logs
