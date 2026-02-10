from pathlib import Path

NAN_THRESHOLD: float = 0.5

ROOTDIR = Path(__file__).parents[1]
DATADIR = ROOTDIR / "data" / "ForbushDecrease"
FIGDIR = ROOTDIR / "figures" / "ForbushDecrease"
MIN_VALUE_THRESHOLD: float = 1e-16

METRIC_COLUMNS = [
    "global_efficiency",  # x1
    "entropy",  # x2
    "hurst_rs",  # x3
    "fractal",  # x4
    "modularity",  # x5
    "assortativity",  # x6
    "estrada_index",  # x7
    "avg_katz",  # x8
    "avg_closeness",  # x9
    "avg_betweenness",  # x10
    "avg_laplacian",  # x11
]
