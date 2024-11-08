"""Public package to get correct model.

Raises:
    ValueError: error if name isn't a model

Returns:
    Model: returns a model that corresponds with the name
"""

from typing import TYPE_CHECKING

from autoop.core.ml.model.classification.k_nearest_neighbors import (
    KNearestNeighbors,
)
from autoop.core.ml.model.classification.linear_svc import LinearSVC
from autoop.core.ml.model.classification.random_forest_classifier import (
    RandomForestClassifier,
)

if TYPE_CHECKING:
    from autoop.core.ml.model.model import Model

from autoop.core.ml.model.regression.lasso import Lasso
from autoop.core.ml.model.regression.multiple_linear_regression import (
    MultipleLinearRegression,
)
from autoop.core.ml.model.regression.ridge import Ridge

REGRESSION_MODELS = [
    "MultipleLinearRegression",
    "Lasso",
    "Ridge",
]  # add your models as str here

CLASSIFICATION_MODELS = [
    "KNearestNeighbors",
    "LinearSVC",
    "RandomForestClassifier",
]  # add your models as str here


def get_model(model_name: str) -> "Model":
    """Get a model by name using this Factory Function."""
    match model_name:
        case "MultipleLinearRegression":
            return MultipleLinearRegression()
        case "Lasso":
            return Lasso()
        case "Ridge":
            return Ridge()
        case "KNearestNeighbors":
            return KNearestNeighbors()
        case "LinearSVC":
            return LinearSVC()
        case "RandomForestClassifier":
            return RandomForestClassifier()
    raise ValueError(f"Model {model_name} doesn't exist.")
