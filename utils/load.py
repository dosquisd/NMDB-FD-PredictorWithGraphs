"""Data loading utilities for neutron monitor data.

This module provides functions to load and process neutron monitor data
from text files into pandas DataFrames.
"""

from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

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

        if len(rows[0]) != len(columns):
            # Remove first column (duplicate datetime)
            rows = list(map(lambda x: x[1:], rows))

    df = pd.DataFrame(rows, columns=columns)
    return df


def load_events() -> Dict[str, EventData]:
    def load_event_metadata(
        file_path: Optional[Path] = None,
    ) -> Dict[str, Dict[str, str]]:
        """Load event metadata from a specified file path.
        Usually the path is `events.csv` in the FD directory.
        """
        if file_path is None or not file_path.exists():
            file_path = DATADIR / "events.csv"

        df = pd.read_csv(file_path, delimiter=",")
        selected_columns = [
            "event_date",
            "drop",
            "geomagnetic_storm_level",
            "Dst_nT",
        ]
        metadata = df[selected_columns]
        metadata_dict = metadata.set_index("event_date").to_dict(orient="index")
        metadata_dict = {
            str(event_date): {
                "drop": float(data["drop"] or 0),
                "intensity": data["geomagnetic_storm_level"] or "Unknown",
                "dst": float(data["Dst_nT"] or 0),
            }
            for event_date, data in metadata_dict.items()
        }
        return metadata_dict

    def load_stations_metadata(
        file_path: Optional[Path] = None,
    ) -> Dict[str, Dict[str, float]]:
        """load station metadata from a specified file path.
        Usually the path is `stations_metadata.csv` in the FD directory.
        """

        if file_path is None or not file_path.exists():
            file_path = DATADIR / "stations_metadata.csv"

        metadata_df = pd.read_csv(file_path).set_index("station")
        selected_columns = ["cutoff_rigidity", "altitude_m"]
        metadata = metadata_df[selected_columns]
        metadata_dict = metadata.to_dict(orient="dict")
        metadata_dict = {
            "cutoff_rigidity": metadata_dict["cutoff_rigidity"],
            "altitude": metadata_dict["altitude_m"],
        }

        return metadata_dict

    event_files = list(DATADIR.glob("*"))
    event_metadata = load_event_metadata()
    station_metadata = load_stations_metadata()

    # Datetime is already parsed to datetime
    events: Dict[str, EventData] = {
        f.name: EventData(
            raw=load_data(f / "all.txt").set_index("datetime"),
            graphs={},
            **station_metadata,  # type: ignore
            **event_metadata.get(f.name, {}),
        )
        for f in event_files
        if f.is_dir()
    }

    return events


if __name__ == "__main__":
    from pprint import pprint

    pprint(load_events())
