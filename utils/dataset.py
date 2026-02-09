from pathlib import Path
from typing import Dict, Optional, TypedDict

import pandas as pd
from sklearn.impute import KNNImputer

from .constants import MIN_VALUE_THRESHOLD
from .dtypes import EventData
from .enums import AdjacencyMethod, DistanceTransformation, Normalizer
from .graph import GraphEvent
from .load import load_events
from .utils import invalid_stations, logger


class FilenameVariables(TypedDict):
    event_filename: str
    imput_data: bool
    use_threshold: bool


def get_events(
    *,
    filename: Optional[str] = None,
    resample_time: str = "5min",
    imput_data: bool = False,
    use_threshold: bool = False,
) -> Dict[str, EventData]:
    if not filename:
        filename = "all.txt"

    events = load_events(filename=filename)

    # Fill data preprocessing
    imputer = KNNImputer(n_neighbors=2, weights="uniform")
    for event_name, data in events.items():
        df = data["raw"]

        # Drop stations with too many NaNs
        stations_to_drop = invalid_stations(df, use_threshold=use_threshold)
        df = df.drop(columns=stations_to_drop)

        # Impute missing values using KNN Imputer
        if imput_data:
            df.loc[:, :] = imputer.fit_transform(df)

        # Resample to 5-minute intervals using median
        events[event_name]["raw"] = df.resample(resample_time).median()
        logger.debug(f"Event {event_name}: Dropped stations {stations_to_drop}")

    return events


def decode_variables_from_filename(filename: Path) -> FilenameVariables:
    # filename format: dataset_{event_filename}_imput-{True|False}_threshold-{True|False}.csv

    # Extract the section after 'dataset_'
    filename_sections = filename.stem.removeprefix("dataset_").split("_")
    # Get the event filename (e.g., 'all', 'all.original', 'all.imp')
    event_filename = f"{filename_sections[0]}.txt"
    # Extract the imput_data value from the filename
    imput_data = filename_sections[1].split("-")[1] == "True"
    # Extract the use_threshold value from the filename
    use_threshold = filename_sections[2].split("-")[1] == "True"

    return {
        "event_filename": event_filename,
        "imput_data": imput_data,
        "use_threshold": use_threshold,
    }


def encode_variables_to_filename(
    event_filename: str, imput_data: bool, use_threshold: bool
) -> str:
    # Create a filename based on the provided variables
    return f"dataset_{event_filename.removesuffix('.txt')}_imput-{imput_data}_threshold-{use_threshold}.csv"


def read_dataset(filename: Path) -> pd.DataFrame:
    # filename format: dataset_{event_filename}_imput-{True|False}_threshold-{True|False}.csv

    variables = decode_variables_from_filename(filename)
    event_filename = variables["event_filename"]
    imput_data = variables["imput_data"]
    use_threshold = variables["use_threshold"]

    # Get the events data based on the extracted filename and options
    events = get_events(
        filename=event_filename,
        imput_data=imput_data,
        use_threshold=use_threshold,
    )

    graphs = []
    df = pd.read_csv(filename)
    for row in df.itertuples():
        event_date = str(row.event_date)
        event_data = events[event_date]

        transformation_value = row.transformation
        normalization_value = row.normalization
        adj_method_value = row.adjacency_method

        transform_method = DistanceTransformation(transformation_value)
        normalization = Normalizer(normalization_value)
        adj_method = AdjacencyMethod(adj_method_value)

        # Create graph event with the raw data and metadata
        raw_df = event_data["raw"].reset_index(drop=True)
        if transform_method == DistanceTransformation.LOG:
            raw_df[raw_df.abs() < MIN_VALUE_THRESHOLD] = MIN_VALUE_THRESHOLD

        transformed_data = transform_method.transform(raw_df.to_numpy())
        transformed_df = pd.DataFrame(transformed_data, columns=raw_df.columns)
        transformed_df[transformed_df.abs() < MIN_VALUE_THRESHOLD] = MIN_VALUE_THRESHOLD

        normalized_data = normalization.normalize(transformed_df.to_numpy())
        normalized_df = pd.DataFrame(normalized_data, columns=raw_df.columns)
        nan_sum = normalized_df.isna().sum()
        nan_columns = nan_sum[nan_sum > 0].index.tolist()
        if nan_columns:
            logger.warning(f"NAN COLUMNS DROPPED: {nan_columns}")
            normalized_df.drop(columns=nan_columns, inplace=True)

        graph = GraphEvent(
            data=normalized_df,
            metadata={
                # Event metadata
                "drop": event_data.get("drop", 0.0),
                "intensity": event_data.get("intensity", "Unknown"),
                "dst": event_data.get("dst", 0.0),
                # Station metadata
                "cutoff_rigidity": event_data["cutoff_rigidity"],
                "altitude": event_data["altitude"],
            },
        ).get_graph_networkx(adj_method, threshold=0.0)

        graphs.append(graph)

    # I know, this is not recommened, but I don't care.
    # I just want to have the graph objects in the dataframe for later use
    df["graph"] = graphs
    return df
