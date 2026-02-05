import enum
import warnings

import numpy as np


class Normalizer(enum.Enum):
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


class DistanceTransformation(enum.Enum):
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


if __name__ == "__main__":
    import pandas as pd

    from .constants import ROOTDIR

    df = pd.read_csv(ROOTDIR / "notebooks" / "raw_df.csv", index_col=0)
    print(df.head())

    transformation = DistanceTransformation.LOG
    transformed_data = transformation.transform(df.to_numpy())
    transformed_df = pd.DataFrame(transformed_data, columns=df.columns)
    print(transformed_df.head())

    assert df.shape == transformed_df.shape
    print("\n\n\n")

    example_df = df.copy()
    example_df["MXCO"] = np.log(example_df["MXCO"] / np.median(example_df["MXCO"]))

    print("Example transformation:")
    print(example_df["MXCO"])

    print("Transformed transformation:")
    print(transformed_df["MXCO"])

    # The same result!!!