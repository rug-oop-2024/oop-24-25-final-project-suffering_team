from autoop.core.ml.model.model import Model

import numpy as np
from sklearn.neighbors import KNeighborsClassifier as SkKNeighborsClassifier


class KNearestNeighbors(Model):
    """A KNearestNeighbors implementation of the Model class."""

    def __init__(self, *args, k_value: int = 3, **kwargs) -> None:
        """Initialize model.

        Args:
            k_value (int): The number of closest neighbors to check.
        """
        super().__init__()
        self.k = k_value
        self._model = SkKNeighborsClassifier(
            *args, n_neighbors=self.k, **kwargs
        )
        # Add hyper parameters to the parameters dictionary using the setter.
        new_parameters = self._model.get_params()
        self.parameters = new_parameters
        self.type = "classification"

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
            k_value (int): The value for k that needs to be checked. Must be
                greater than 0.

        Raises:
            TypeError: If k_value is no an integer.
            ValueError: If k not larger than 0.

        Returns:
            int: The checked value of k.
        """
        if not isinstance(k_value, int):
            raise TypeError("k must be an integer")
        if k_value <= 0:
            raise ValueError("k must be greater than 0")
        return k_value

    def fit(self, observations: np.ndarray, ground_truths: np.ndarray) -> None:
        """Fit the KNN model.

        Args:
            observations (np.ndarray): Observations used to train the model.
                Row dimension is samples, column dimension is variables.
            ground_truths (np.ndarray): Ground_truths corresponding to the
                observations used to train the model. Row dimension is samples.
        """
        self._check_fit_requirements(observations, ground_truths)

        if self.k > observations.shape[0]:
            raise ValueError(
                f"k ({self.k}) cannot be greater than the number of "
                f"observations ({observations.shape[0]})."
            )

        # If the ground truth is one-hot-encoded: extract label indices
        if ground_truths.ndim > 1:
            ground_truths = np.argmax(ground_truths, axis=1)

        # Train the model
        self._model.fit(observations, ground_truths)

        self._fitted = True
        self._n_features = observations.shape[1]

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Use the model to predict values for observations.

        Args:
            observations (np.ndarray): The observations which need predictions.
                Row dimension is samples, column dimension is variables.

        Returns:
            np.ndarray: The classification of the observation.
        """
        self._check_predict_requirements(observations)
        return self._model.predict(observations)
