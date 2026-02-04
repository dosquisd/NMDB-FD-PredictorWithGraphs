"""Data loading utilities for neutron monitor data.

This module provides functions to load and process neutron monitor data
from text files into pandas DataFrames.
"""

from functools import lru_cache
from pathlib import Path
from typing import Dict

import pandas as pd

from .constants import DATADIR
from .dtypes import EventData


@lru_cache(maxsize=None)
def load_data(file_path: str) -> pd.DataFrame:
    """Load and read neutron monitor data from a specified file path.

    Reads data from a text file containing neutron monitor measurements
    and converts it into a pandas DataFrame. The file should be in the
    format used by ./data/ForbushDecrease or similar directories.

    Args:
        file_path (str): Path to the text file containing the data.

    Returns:
        A pandas DataFrame with datetime index and station data columns.

    Raises:
        FileNotFoundError: If the specified file path does not exist.
        ValueError: If the file format is invalid or cannot be parsed.
    """

    def clean_row(row: str) -> list[pd.Timestamp | float | None]:
        """Process a single row of data, converting values appropriately.

        Converts row values to appropriate types: datetime for the first column,
        float for numeric values, and None for 'null' entries.

        Args:
            row: A single row of data as a string.

        Returns:
            A list of processed values with appropriate types.

        Note:
            All rows must have the same format: first column is datetime,
            subsequent columns are float values or 'null'.
        """
        values = row.strip().split(";")
        cleaned_values = []
        for value in values:
            value = value.strip()

            # Append None values
            if value.lower() == "null":
                cleaned_values.append(None)
                continue

            # Parse float values
            try:
                cleaned_values.append(float(value))
                continue
            except ValueError:
                cleaned_values.append(value)

            # Parse datetime values
            try:
                cleaned_values.append(pd.to_datetime(value, format="%Y-%m-%d %H:%M:%S"))
                continue
            except Exception:  # ValueError, DateParseError
                pass

        return cleaned_values

    with open(file_path, "r") as file:
        lines = file.readlines()
        header = lines[0].strip().split("   ")
        columns = ["datetime"] + list(map(lambda x: x.strip(), header))
        rows = list(map(clean_row, lines[1:]))

        if rows[0] != len(columns):
            rows = list(
                map(lambda x: x[1:], rows)
            )  # Remove first column (duplicate datetime)

    df = pd.DataFrame(rows, columns=columns)
    return df


def load_events() -> Dict[str, EventData]:
    def load_intensity(file_path: Path) -> str:
        with open(file_path, "r") as f:
            intensity = f.read().strip()
        return intensity

    def load_cutoff_rigidity(file_path: Path) -> Dict[str, float]:
        if not file_path.exists():
            return {}

        metadata = pd.read_csv(file_path)
        cutoff_rigidity = {
            row["station"]: row["cutoff_rigidity"] for _, row in metadata.iterrows()
        }
        return cutoff_rigidity

    event_files = list(DATADIR.glob("*"))

    # Datetime is already parsed to datetime
    events: Dict[str, EventData] = {
        f.name: EventData(
            raw=load_data(f / "all.txt").set_index("datetime"),
            intensity=load_intensity(f / "intensity.txt"),
            graphs={},
            cutoff_rigidity=load_cutoff_rigidity(f / "stations_metadata.csv"),
        )
        for f in event_files
        if f.is_dir()
    }

    return events
