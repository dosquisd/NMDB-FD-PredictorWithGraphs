import warnings
from enum import Enum
from typing import Any, Callable, Dict, Union

import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial.distance import minkowski
from sklearn.metrics.pairwise import cosine_distances, manhattan_distances

from .constants import MIN_VALUE_THRESHOLD

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


class Normalizer(Enum):
    MIN_MAX = "min_max"
    Z_SCORE = "z_score"
    ROBUST = "robust"
    DECIMAL_SCALING = "decimal_scaling"
    NONE = "none"

    def __min_max_normalize(self, data: np.ndarray) -> np.ndarray:
        """Applies Min-Max normalization to the data."""
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        normalized_data = (data - min_val) / (max_val - min_val)
        return normalized_data

    def __z_score_normalize(self, data: np.ndarray) -> np.ndarray:
        """Applies Z-score normalization to the data."""
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        normalized_data = (data - mean) / std
        return normalized_data

    def __robust_normalize(self, data: np.ndarray) -> np.ndarray:
        """Applies Robust normalization to the data."""
        median = np.median(data, axis=0)
        q1 = np.percentile(data, 25, axis=0)
        q3 = np.percentile(data, 75, axis=0)
        iqr = q3 - q1
        iqr[np.abs(iqr) < MIN_VALUE_THRESHOLD] = MIN_VALUE_THRESHOLD

        normalized_data = (data - median) / iqr
        return normalized_data

    def __decimal_scaling_normalize(self, data: np.ndarray) -> np.ndarray:
        """Applies Decimal Scaling normalization to the data."""
        max_abs = np.max(np.abs(data), axis=0)
        j = np.ceil(np.log10(max_abs + 1))
        normalized_data = data / (10**j)
        return normalized_data

    def normalize(self, data: np.ndarray) -> np.ndarray:
        match self:
            case Normalizer.MIN_MAX:
                return self.__min_max_normalize(data)
            case Normalizer.Z_SCORE:
                return self.__z_score_normalize(data)
            case Normalizer.ROBUST:
                return self.__robust_normalize(data)
            case Normalizer.DECIMAL_SCALING:
                return self.__decimal_scaling_normalize(data)
            case Normalizer.NONE:
                return data
            case _:
                warnings.warn(
                    f"Unsupported normalization method: {self.value}. Returning original data."
                )
                return data


class DistanceTransformation(Enum):
    NONE = "none"
    LOG = "log"
    EXPONENTIAL = "exponential"

    def __log_transform(self, data: np.ndarray) -> np.ndarray:
        """Applies logarithmic transformation to the data."""
        median = np.median(data, axis=0)
        transformed_data = np.log(data / median)
        return transformed_data

    def __exponential_transform(self, data: np.ndarray) -> np.ndarray:
        """Applies exponential transformation to the data."""
        median = np.median(data, axis=0)
        transformed_data = np.exp(data / median)
        return transformed_data

    def transform(self, data: np.ndarray) -> np.ndarray:
        match self:
            case DistanceTransformation.LOG:
                return self.__log_transform(data)
            case DistanceTransformation.EXPONENTIAL:
                return self.__exponential_transform(data)
            case DistanceTransformation.NONE:
                return data
            case _:
                warnings.warn(
                    f"Unsupported distance transformation method: {self.value}. Returning original data."
                )
                return data
