import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Union

import torch.distributed as dist

Pathlike = Union[str, Path]

_LOGGER_NAME = "app"
_LOGGER_INITIALIZED = False


def setup_logger(
    log_dir: Pathlike,
    log_level: str = "info",
    use_console: bool = True,
) -> logging.Logger:
    """
    Initialize global application logger ONCE.
    All modules should reuse this logger.
    """
    global _LOGGER_INITIALIZED

    logger = logging.getLogger(_LOGGER_NAME)

    # ⭐ already initialized → just return
    if _LOGGER_INITIALIZED:
        return logger

    # --------------------------------------------------
    # rank / world size
    # --------------------------------------------------
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        fmt = (
            "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] "
            f"({rank}/{world_size}) %(message)s"
        )
    else:
        rank = 0
        fmt = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    # --------------------------------------------------
    # log file path (ONLY ONCE)
    # --------------------------------------------------
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    timestamp = now.strftime("%H-%M-%S")

    log_filename = f"{log_dir}/{date}/{timestamp}"
    if rank != 0:
        log_filename = f"{log_filename}-{rank}"

    os.makedirs(os.path.dirname(log_filename), exist_ok=True)

    # --------------------------------------------------
    # log level
    # --------------------------------------------------
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    level = level_map.get(log_level.lower(), logging.INFO)

    # --------------------------------------------------
    # configure logger
    # --------------------------------------------------
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(fmt)

    # --------------------------------------------------
    # file handler (ONE FILE)
    # --------------------------------------------------
    file_handler = logging.FileHandler(log_filename, mode="w")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(lambda r: r.levelno != logging.WARNING)
    logger.addHandler(file_handler)

    # --------------------------------------------------
    # console handler (rank 0)
    # --------------------------------------------------
    if rank == 0 and use_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(lambda r: r.levelno != logging.WARNING)
        logger.addHandler(console_handler)

    _LOGGER_INITIALIZED = True
    return logger