import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

from utils.constants import DATADIR, FIGDIR, MIN_VALUE_THRESHOLD, NAN_THRESHOLD, ROOTDIR
from utils.dtypes import AdjacencyMethod, EventData, Graph
from utils.graph import GraphEvent
from utils.load import load_data, load_events
from utils.metrics import box_covering_cbb, graph_fractal_dimension
from utils.normalize import DistanceTransformation, Normalizer


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
    "setup_plotting",
]
