import logging
import os
import sys
from .. import __version__


FMT = "%(asctime)s - %(name)s - %(module)s - %(levelname)s - %(message)s"
# logger = logging.getLogger(__name__)


def setuplog(
    name: str = "pyorc",
    path: str = None,
    log_level: int = 20,
    fmt: str = FMT,
    append: bool = True,
) -> logging.Logger:
    f"""Set-up the logging on sys.stdout and file if path is given.
    Parameters
    ----------
    name : str, optional
        logger name, by default "hydromt"
    path : str, optional
        path to logfile, by default None
    log_level : int, optional
        Log level [0-50], by default 20 (info)
    fmt : str, optional
        log message formatter, by default {FMT}
    append : bool, optional
        Wether to append (True) or overwrite (False) to a logfile at path, by default True
    Returns
    -------
    logging.Logger
        _description_
    """
    logger = logging.getLogger(name)
    for _ in range(len(logger.handlers)):
        logger.handlers.pop().close()  # remove and close existing handlers
    logging.captureWarnings(True)
    logger.setLevel(log_level)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(log_level)
    console.setFormatter(logging.Formatter(fmt))
    logger.addHandler(console)
    if path is not None:
        if append is False and os.path.isfile(path):
            os.unlink(path)
        add_filehandler(logger, path, log_level=log_level, fmt=fmt)
    logger.info(f"pyorc version: {__version__}")

    return logger


def add_filehandler(logger, path, log_level=20, fmt=FMT):
    """Add file handler to logger."""
    if not os.path.isdir(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    isfile = os.path.isfile(path)
    ch = logging.FileHandler(path)
    ch.setFormatter(logging.Formatter(fmt))
    ch.setLevel(log_level)
    logger.addHandler(ch)
    if isfile:
        logger.debug(f"Appending log messages to file {path}.")
    else:
        logger.debug(f"Writing log messages to new file {path}.")
