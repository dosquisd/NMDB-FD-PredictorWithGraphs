from typing import Dict, TypedDict

import pandas as pd

from .enums import AdjacencyMethod, Graph


class EventData(TypedDict):
    raw: pd.DataFrame
    graphs: Dict[AdjacencyMethod, Graph]
    cutoff_rigidity: Dict[str, float]
    altitude: Dict[str, float]
    intensity: str
    drop: float
    dst: float
