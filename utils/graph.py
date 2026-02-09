from typing import Any, Dict, Literal

import igraph as ig
import networkx as nx
import pandas as pd

from .enums import AdjacencyMethod, Graph


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
