from autoop.core.ml.model.classification.k_nearest_neighbors import (
    KNearestNeighbors
)
from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression.lasso import Lasso
from autoop.core.ml.model.regression.multiple_linear_regression import (
    MultipleLinearRegression,
)

REGRESSION_MODELS = [
    "MultipleLinearRegression",
    "Lasso",
]  # add your models as str here

CLASSIFICATION_MODELS = ["KNearestNeighbors"]  # add your models as str here


def get_model(model_name: str) -> Model:
    """Factory function to get a model by name."""
    raise NotImplementedError("To be implemented.")
