from typing import List

from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature


def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """Assumption: only categorical and numerical features and no NaN values.

        dataset: Dataset
    Returns:
        List[Feature]: List of features with their types.
    """
    features = []
    data = dataset.read()
    columns = data.columns.tolist()
    for column in columns:
        name = column
        values = data[column]

        try:
            float(values[0])
            column_type = "numerical"
        except ValueError:
            column_type = "categorical"
        feature = Feature(name, column_type)
        features.append(feature)
    return features
