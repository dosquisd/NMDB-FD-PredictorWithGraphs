from enum import Enum
from typing import Any, Callable, Dict, TypedDict, Union

import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial.distance import minkowski
from sklearn.metrics.pairwise import cosine_distances, manhattan_distances

Graph = Union[ig.Graph, nx.Graph]


class AdjacencyMethod(Enum):
    CORRELATION = "correlation"
    COSINE = "cosine"
    MANHATTAN = "manhattan"
    MINKOWSKI = "minkowski"

    def __correlation_func(self, df: pd.DataFrame, _) -> pd.DataFrame:
        corr = df.corr()
        columns = corr.columns.tolist()
        adjacency_matrix = pd.DataFrame(
            data=np.sqrt(2 * (1 - corr)),
            index=columns,
            columns=columns,
            dtype=float,
        )

        return adjacency_matrix

    def __cosine_func(self, df: pd.DataFrame, _) -> pd.DataFrame:
        columns = df.columns.tolist()
        adjacency_matrix = pd.DataFrame(
            data=cosine_distances(df.T),
            index=columns,
            columns=columns,
            dtype=float,
        )

        return adjacency_matrix

    def __manhattan_func(self, df: pd.DataFrame, _) -> pd.DataFrame:
        columns = df.columns.tolist()
        adjacency_matrix = pd.DataFrame(
            data=manhattan_distances(df.T),
            index=columns,
            columns=columns,
            dtype=float,
        )
        return adjacency_matrix

    def __minkowski_func(
        self, df: pd.DataFrame, kwargs: Dict[str, Any]
    ) -> pd.DataFrame:
        p = kwargs.get("p", 3)
        try:
            p = int(p)
        except ValueError:
            p = 3

        kwargs.pop("p", None)

        # Compute pairwise Minkowski distances
        # Scipy does not have a built-in function for pairwise Minkowski distances,
        # so we compute it manually. This is less efficient than the other methods tbh.
        columns = df.columns.tolist()
        shape = (df.shape[1], df.shape[1])
        data = np.zeros(shape)
        for i, col_i in enumerate(columns):
            for j, col_j in enumerate(columns):
                if i > j:
                    data[i, j] = data[j, i]
                    continue

                # i < j case is always computed before i > j
                # this is a symmetric matrix
                data[i, j] = minkowski(
                    df[col_i].to_numpy(), df[col_j].to_numpy(), p=p, **kwargs
                )

        adjacency_matrix = pd.DataFrame(
            data=data,
            index=columns,
            columns=columns,
            dtype=float,
        )
        return adjacency_matrix

    def get_function(self) -> Callable[[pd.DataFrame, Any], pd.DataFrame]:
        match self.name:
            case "CORRELATION":
                return self.__correlation_func
            case "COSINE":
                return self.__cosine_func
            case "MANHATTAN":
                return self.__manhattan_func
            case "MINKOWSKI":
                return self.__minkowski_func
            case _:
                raise NotImplementedError(
                    f"Adjacency method {self.name} not implemented."
                )


class EventData(TypedDict):
    raw: pd.DataFrame
    graphs: Dict[AdjacencyMethod, Graph]
    cutoff_rigidity: Dict[str, float]
    altitude: Dict[str, float]
    intensity: str
