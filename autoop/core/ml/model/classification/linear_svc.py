from autoop.core.ml.model.model import Model

import numpy as np
from sklearn.svm import LinearSVC as SkLinearSVC


class LinearSVC(Model):
    """A Lasso implementation of the Model class."""

    def __init__(self, *args, **kwargs):
        """Initialize the LinearSVC model with the provided parameters.

        Args:
            *args: Positional arguments for LinearSVC's parameters.
            **kwargs: Keyword arguments for LinearSVC's parameters.
        """
        super().__init__()
        self._model = SkLinearSVC(*args, **kwargs)
        # Add hyper parameters to the parameters dictionary using the setter.
        new_parameters = self._model.get_params()
        self.parameters = new_parameters
        self.type = "classification"

    def fit(self, observations: np.ndarray, ground_truths: np.ndarray) -> None:
        """Use the observations and ground_truths to train the LinearSVC model.

        Args:
            observations (np.ndarray): Observations used to train the model.
                Row dimension is samples, column dimension is variables.
            ground_truths (np.ndarray): Ground_truths corresponding to the
                observations used to train the model. Row dimension is samples.

        Raises:
            ValueError:
                The number of observations and ground_truths must be equal.
            ValueError:
                At least two observations are needed for regression.
        """
        observation_rows = observations.shape[0]
        ground_truth_rows = ground_truths.shape[0]
        if observation_rows != ground_truth_rows:
            raise ValueError(
                f"The number of observations ({observation_rows}) and ",
                f"ground_truths ({ground_truth_rows}) ",
                "should be equal.",
            )

        # If the ground truth is one-hot-encoded: extract label indices
        if ground_truths.ndim > 1:
            ground_truths = np.argmax(ground_truths, axis=1)

        # Train the model
        self._model.fit(observations, ground_truths)

        # Add the coefficients and intercept to parameters using the setter.
        self.parameters = {
            "coefficients": np.array(self._model.coef_),
            "intercept": np.atleast_1d(self._model.intercept_),
        }

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Use the model to predict values for observations.

        Args:
            observations (np.ndarray): The observations which need predictions.
                Row dimension is samples, column dimension is variables.

        Raises:
            ValueError: There are no parameters, the model needs to be fitted
                first.

        Returns:
            list: Predicted values for the observations.
        """
        if "coefficients" not in self.parameters:
            raise ValueError(
                "Model not fitted. Call 'fit' with appropriate arguments"
                "before using 'predict'"
            )

        return self._model.predict(observations)
