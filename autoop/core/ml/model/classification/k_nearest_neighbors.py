from collections import Counter

import numpy as np
from pydantic import BaseModel, Field, field_validator

from autoop.core.ml.model.model import Model


class KNearestNeighbors(Model, BaseModel):
    """A KNearestNeighbors implementation of the Model class."""

    k: int = Field(default=3)

    def __init__(self):
        """Initialize model."""
        super().__init__()

    @field_validator("k")
    def k_greater_than_zero(cls, value: int) -> int:
        """Validate the k field.

        Args:
            value (int):
                The value for k that needs to be checked. Must be greater
                than 0.

        Raises:
            TypeError:
                K must be an integer
            ValueError:
                K must be greater than 0

        Returns:
            int:
                The checked value of k.
        """
        if not isinstance(value, int):
            raise TypeError("k must be an integer")
        if value <= 0:
            raise ValueError("k must be greater than 0")
        return value

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """Store the observations and ground_truths in a dictionary.

        Args:
            observations (np.ndarray):
                Observations used to train the model. Row dimension is
                samples, column dimension is variables.
            ground_truths (np.ndarray):
                Ground_truths corresponding to theobservations used to
                train the model. Row dimension is samples.

        Raises:
            ValueError:
                The number of ground_truths and observations should be equal.
        """
        observation_rows = observations.shape[0]
        ground_truth_rows = ground_truth.shape[0]
        if observation_rows != ground_truth_rows:
            raise ValueError(
                "The number of observations and ground_truths "
                "should be the equal."
            )
        self.parameters = {
            "observations": observations,
            "ground_truth": ground_truth,
        }

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Predict for each observation how it should be classified.

        Args:
            observations (np.ndarray):
                The observation for which require classification. Row
                dimension is samples, column dimension is variables.

        Raises:
            ValueError:
                There are no parameters, the model needs to be fitted first.
            ValueError:
                K should not be larger than the number of observations.

        Returns:
            np.ndarray:
                The classifications of the observations.
        """
        if (
            "observations" not in self.parameters.keys()
            or "ground_truth" not in self.parameters.keys()
        ):
            raise ValueError(
                "Model not fitted. Call 'fit' with appropriate arguments "
                "before using 'predict'"
            )
        if self.k > observations.shape[0]:
            raise ValueError("k is larger than than the number of observations")

        predictions = [
            self._predict_single(observation) for observation in observations
        ]
        return np.array(predictions)

    def _predict_single(self, observation: np.ndarray) -> int:
        """Predict for a single observation how it should be classified.

        This method looks at the labels of the k nearest neighbors and
        chooses the label that is most common.

        Args:
            observations (np.ndarray):
                The observation that needs to be classified.

        Returns:
            int:
                The classification of the observation.
        """
        # Calculate the distance to each point
        distances = np.linalg.norm(
            observation - self.parameters["observations"], axis=1
        )

        # Sort and return the most common label in the k nearest points
        sorted_indices = np.argsort(distances)
        k_neighbors_indices = sorted_indices[: self.k]
        k_neighbors_nearest_labels = [
            self.parameters["ground_truth"][i] for i in k_neighbors_indices
        ]
        most_common = Counter(k_neighbors_nearest_labels).most_common(1)
        return most_common[0][0]
