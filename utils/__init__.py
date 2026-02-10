import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

from utils.constants import (
    DATADIR,
    FIGDIR,
    METRIC_COLUMNS,
    MIN_VALUE_THRESHOLD,
    NAN_THRESHOLD,
    ROOTDIR,
)
from utils.dataset import (
    decode_variables_from_filename,
    encode_variables_to_filename,
    get_dataset_filename,
    get_events,
    read_dataset,
)
from utils.dtypes import EventData
from utils.enums import AdjacencyMethod, DistanceTransformation, Graph, Normalizer
from utils.graph import GraphEvent
from utils.load import load_data, load_events
from utils.metrics import box_covering_cbb, graph_fractal_dimension
from utils.utils import invalid_stations, logger, setup_logger


# LaTeX must be installed previously for this to work
def setup_plotting():
    plt.style.use(["science", "nature"])
    plt.rcParams.update(
        {
            "font.size": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.labelsize": 12,
            "legend.fontsize": 12,
        }
    )


__all__ = [
    "DATADIR",
    "FIGDIR",
    "MIN_VALUE_THRESHOLD",
    "NAN_THRESHOLD",
    "ROOTDIR",
    "get_dataset_filename",
    "get_events",
    "decode_variables_from_filename",
    "encode_variables_to_filename",
    "read_dataset",
    "Graph",
    "GraphEvent",
    "AdjacencyMethod",
    "EventData",
    "load_data",
    "load_events",
    "DistanceTransformation",
    "box_covering_cbb",
    "graph_fractal_dimension",
    "Normalizer",
    "invalid_stations",
    "logger",
    "setup_logger",
    "setup_plotting",
]
