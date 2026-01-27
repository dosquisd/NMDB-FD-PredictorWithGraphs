from enum import Enum
from typing import Any, Callable, Dict, Literal, Union

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


class GraphEvent:
    def __init__(
        self,
        data: pd.DataFrame,
        metadata: Dict[str, Any],
    ) -> None:
        self.data = data
        self.metadata = metadata

    def __get_adjacency_matrix(
        self, method: AdjacencyMethod, **kwargs: Any
    ) -> pd.DataFrame:
        method_func = method.get_function()
        adjacency_matrix = method_func(self.data, kwargs)
        return adjacency_matrix

    def get_graph_networkx(
        self,
        method: AdjacencyMethod,
        **kwargs: Any,
    ) -> nx.Graph:
        adjacency_matrix = self.__get_adjacency_matrix(method, **kwargs)
        graph = nx.from_pandas_adjacency(adjacency_matrix)

        # Set metadata
        for key, value in self.metadata.items():
            graph.graph["metadata_" + key] = value

        return graph

    def get_graph_igraph(
        self,
        method: AdjacencyMethod,
        **kwargs: Any,
    ) -> ig.Graph:
        adjacency_matrix = self.__get_adjacency_matrix(method, **kwargs)
        values = adjacency_matrix.values
        graph: ig.Graph = ig.Graph.Weighted_Adjacency(
            matrix=values.tolist(),
            mode="undirected",
            attr="weight",
        )

        # Set vertex names and metadata
        graph.vs["name"] = adjacency_matrix.columns.tolist()
        for key, value in self.metadata.items():
            graph["metadata_" + key] = value

        return graph

    def get_graph(
        self,
        method: AdjacencyMethod,
        library: Literal["networkx", "igraph"] = "igraph",
    ) -> Graph:
        match library:
            case "networkx":
                return self.get_graph_networkx(method)  # type: ignore
            case "igraph":
                return self.get_graph_igraph(method)
            case _:
                raise NotImplementedError(f"Graph library {library} not implemented.")
