from autoop.core.ml.model.model import Model

import numpy as np
from sklearn.linear_model import LinearRegression as SkLinearRegression


class MultipleLinearRegression(Model):
    """A MultipleLinearRegression implementation of the model class."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the MultipleLinearRegression model."""
        super().__init__()
        self._model = SkLinearRegression(*args, **kwargs)
        # Add hyper parameters to the parameters dictionary using the setter.
        new_parameters = self._model.get_params()
        self.parameters = new_parameters
        self.type = "regression"

    def fit(self, observations: np.ndarray, ground_truths: np.ndarray) -> None:
        """Train the model using observations and ground_truths.

        Args:
            observations (np.ndarray): Observations used to train the model.
                Row dimension is samples, column dimension is variables.
            ground_truths (np.ndarray): Ground_truths corresponding to the
                observations used to train the model. Row dimension is samples.
        """
        self._check_fit_requirements(observations, ground_truths)

        # Train the model
        self._model.fit(observations, ground_truths)

        # Add the coefficients and intercept to parameters using the setter.
        self.parameters = {
            "coefficients": np.array(self._model.coef_),
            "intercept": np.atleast_1d(self._model.intercept_),
        }
        self._fitted = True
        self._n_features = observations.shape[1]

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Use the model to predict values for observations.

        Args:
            observations (np.ndarray): The observations which need predictions.
                Row dimension is samples, column dimension is variables.

        Returns:
            list: Predicted values for the observations.
        """
        self._check_predict_requirements(observations)
        return self._model.predict(observations)
