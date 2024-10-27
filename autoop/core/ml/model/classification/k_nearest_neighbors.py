from autoop.core.ml.model.model import Model

import numpy as np


class KNearestNeighbors(Model):
    """A KNearestNeighbors implementation of the Model class."""

    def __init__(self, k_value: int = 3):
        """Initialize model."""
        super().__init__()
        self.type = "classification"
        self.k = k_value

    @property
    def k(self) -> int:
        """Get the value of k."""
        return self._k

    @k.setter
    def k(self, value: int) -> None:
        """Set the value of k with validation."""
        self._k = self._validate_k(value)

    def _validate_k(self, k_value: int) -> int:
        """Validate the k attribute.

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
        if not isinstance(k_value, int):
            raise TypeError("k must be an integer")
        if k_value <= 0:
            raise ValueError("k must be greater than 0")
        return k_value

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
                f"The number of observations ({observation_rows}) and ",
                f"ground_truths ({ground_truth_rows}) ",
                "should be equal.",
            )

        # If the ground truth is one-hot-encoded: extract label indices
        if ground_truth.ndim > 1:
            ground_truth = np.argmax(ground_truth, axis=1)

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
                If the model has not been fitted.
            ValueError:
                If K is larger than the number of observations.

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
            raise ValueError(
                "k is larger than than the number of observations"
            )

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

        # Find and store the k nearest labels
        sorted_indices = np.argsort(distances)
        k_neighbors_indices = sorted_indices[: self.k]
        k_neighbors_labels = self.parameters["ground_truth"][
            k_neighbors_indices
        ]

        # Find the occurrences of each unique label
        unique_labels, counts = np.unique(
            k_neighbors_labels, return_counts=True
        )

        # Find and return the label which has to most occurrences
        max_index = np.argmax(counts)
        return unique_labels[max_index]
