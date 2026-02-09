import logging
from typing import List

import pandas as pd

from .constants import NAN_THRESHOLD


def invalid_stations(station_df: pd.DataFrame, use_threshold: bool) -> List[str]:
    """Check if a station has less than NAN_THRESHOLD percentage of NaN values."""
    nan_ratios = station_df.isna().mean()

    if use_threshold:
        invalid_stations = nan_ratios[nan_ratios >= NAN_THRESHOLD].index.tolist()
    else:
        invalid_stations = nan_ratios[nan_ratios > 0].index.tolist()

    return invalid_stations


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("NMDB-FD-PredictorWithGraphs")
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(ch)

    return logger


logger = setup_logger()


if __name__ == "__main__":
    logger.info("This is a test log message from utils.py")
