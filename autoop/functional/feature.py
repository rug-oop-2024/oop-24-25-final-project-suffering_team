from typing import List
from pandas.api.types import is_numeric_dtype

from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature


def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """Assumption: only categorical and numerical features and no NaN values.

    Args:
        dataset (Dataset): The dataset with features.
    Returns:
        List[Feature]: List of features with their types.
    """
    features = []
    data = dataset.read()
    columns = data.columns.tolist()
    for column in columns:
        name = column
        values = data[column]
        if is_numeric_dtype(values):
            float(values[0])
            column_type = "numerical"
        else:
            column_type = "categorical"
        feature = Feature(name, column_type)
        features.append(feature)
    return features
