"""Structured terminal logging helpers for phase-based agent flows."""

from __future__ import annotations

import json
import logging
from typing import Any

LOGGER_NAME = "bank_support_app"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a configured application logger suitable for terminal output."""

    logger_name = LOGGER_NAME if name is None else f"{LOGGER_NAME}.{name}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def log_phase(logger: logging.Logger, title: str) -> None:
    """Emit a formatted phase banner."""

    logger.info("================")
    logger.info(title)
    logger.info("================")


def log_json(logger: logging.Logger, label: str, payload: Any) -> None:
    """Pretty-print JSON payloads for local debugging."""

    if hasattr(payload, "model_dump"):
        payload = payload.model_dump()
    logger.info(label)
    logger.info("json")
    logger.info(json.dumps(payload, indent=2, ensure_ascii=False))
