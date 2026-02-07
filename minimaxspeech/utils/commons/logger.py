import logging
import os
from typing import Union

import torch.distributed as dist
from datetime import datetime
from pathlib import Path

Pathlike = Union[str, Path]


def setup_logger(log_dir: Pathlike, log_level: str = "info", use_console: bool = True) -> None:
    """Setup log level.
    Args:
        log_filename:
            The filename to save the log.
        log_level:
            The log level to use, e.g., "debug", "info", "warning", "error",
            "critical"
        use_console:
            True to also print logs to console.
    """
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    timestamp = now.strftime("%H-%M-%S")
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        formatter = f"%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] ({rank}/{world_size}) %(message)s"
        log_filename = f"{log_dir}/{date}/{timestamp}-{rank}"
    else:
        rank = 0
        formatter = f"%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
        log_filename = f"{log_dir}/{date}/{timestamp}"
    
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)

    level = logging.ERROR
    if log_level == "debug":
        level = logging.DEBUG
    elif log_level == "info":
        level = logging.INFO
    elif log_level == "warning":
        level = logging.WARNING
    elif log_level == "critical":
        level = logging.CRITICAL
    
    logging.basicConfig(filename=log_filename, format=formatter, level=level, filemode="w", force=True)
    
    logging.getLogger().handlers[0].addFilter(lambda record: record.levelno != logging.WARNING)
    if rank == 0 and use_console:
        console = logging.StreamHandler()
        console.setLevel(level)
        console.setFormatter(logging.Formatter(formatter))
        logging.getLogger("").addHandler(console)
        logging.getLogger().handlers[1].addFilter(
            lambda record: record.levelno != logging.WARNING
        )