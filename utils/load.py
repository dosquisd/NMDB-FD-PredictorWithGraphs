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
                pass

            # Parse datetime values
            try:
                cleaned_values.append(pd.to_datetime(value, format="%Y-%m-%d %H:%M:%S"))
                continue
            except Exception:  # ValueError, DateParseError
                pass

            # Strip function it's duplicated, but I want to be sure to remove any extra spaces
            if len(value.strip()) > 1:
                cleaned_values.append(value)

        return cleaned_values

    with open(file_path, "r") as file:
        lines = file.readlines()

    # Verify if we can use pandas.read_csv directly
    for delim in (",", ";", "   "):
        # If the delimiter is not in the first line (columns) or is not in the first row,
        # then it is heterogeneous. We can compare each row, but I will only use the
        # first row for simplicity.
        if not (delim in lines[0] and delim in lines[1]):
            continue

        # Now, verify if we can split and get the same amount of data
        columns = lines[0].strip().split(delim)
        first_row = lines[1].strip().split(delim)
        if len(columns) != len(first_row):
            continue

        try:
            df = pd.read_csv(file_path, delimiter=delim)
            df.rename(columns={"DATETIME": "datetime"}, inplace=True)
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])

            return df
        except Exception:
            break

    header = lines[0].strip().split("   ")
    columns = ["datetime"] + list(map(lambda x: x.strip(), header))
    rows = list(map(clean_row, lines[1:]))
    rows = list(filter(lambda x: len(x) > 1, rows))  # Filter out empty rows

    if len(rows[0]) - 1 == len(columns):
        # Remove first column (duplicate datetime)
        rows = list(map(lambda x: x[1:], rows))

    df = pd.DataFrame(rows, columns=columns)
    return df


def load_events(*, filename: str = "all.txt") -> Dict[str, EventData]:
    def load_event_metadata(
        file_path: Optional[Path] = None,
    ) -> Dict[str, Dict[str, str]]:
        """Load event metadata from a specified file path.
        Usually the path is `events.csv` in the FD directory.
        """
        if file_path is None or not file_path.exists():
            file_path = DATADIR / "events.csv"

        df = pd.read_csv(file_path, delimiter=",")
        df.replace(
            ["G5/G4", "G2/G1", "G4/G3", "G3/G2"],
            ["G4/G5", "G1/G2", "G3/G4", "G2/G3"],
            inplace=True,
        )
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
            raw=load_data(
                path if (path := f / filename).exists() else f / "all.original.txt"
            ).set_index("datetime"),
            graphs={},
            **station_metadata,  # type: ignore
            **event_metadata.get(f.name, {}),
        )
        for f in event_files
        if f.is_dir()
    }

    return events
