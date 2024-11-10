from autoop.core.ml.model.model import Model

import numpy as np
from sklearn.ensemble import RandomForestClassifier as SkRandomForestClassifier


class RandomForestClassifier(Model):
    """A RandomForestClassifier implementation of the Model class."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the LinearSVC model with the provided parameters.

        Args:
            *args: Positional arguments for RandomForestClassifier parameters.
            **kwargs: Keyword arguments for RandomForestClassifier parameters.
        """
        super().__init__()
        self._model = SkRandomForestClassifier(*args, **kwargs)
        # Add hyper parameters to the parameters dictionary using the setter.
        new_parameters = self._model.get_params()
        self.parameters = new_parameters
        self.type = "classification"

    def fit(self, observations: np.ndarray, ground_truths: np.ndarray) -> None:
        """Train the RandomForestClassifier model.

        Args:
            observations (np.ndarray): Observations used to train the model.
                Row dimension is samples, column dimension is variables.
            ground_truths (np.ndarray): Ground_truths corresponding to the
                observations used to train the model. Row dimension is samples.
        """
        self._check_fit_requirements(observations, ground_truths)

        # If the ground truth is one-hot-encoded: extract label indices
        if ground_truths.ndim > 1:
            ground_truths = np.argmax(ground_truths, axis=1)

        # Train the model
        self._model.fit(observations, ground_truths)

        self.parameters = {"estimators": np.array(self._model.estimators_)}

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
