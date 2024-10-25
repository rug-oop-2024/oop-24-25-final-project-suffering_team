import numpy as np
from sklearn.linear_model import Lasso as SkLasso

from autoop.core.ml.model.model import Model


class Lasso(Model):
    """A Lasso implementation of the Model class."""

    def __init__(self, *args, **kwargs):
        """Initialize the lasso model with the provided parameters.

        Args:
            *args: Positional arguments for Lasso's parameters.
            **kwargs: Keyword arguments for Lasso's parameters.
        """
        super().__init__()
        self._model = SkLasso(*args, **kwargs)
        # Add hyper parameters to the parameters dictionary using the setter.
        new_parameters = self._model.get_params()
        self.parameters = new_parameters
        self.type = "regression"

    def fit(self, observations: np.ndarray, ground_truths: np.ndarray) -> None:
        """Use the observations and ground_truths to train the Lasso model.

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
        # Verify the input
        if observations.shape[0] != ground_truths.shape[0]:
            raise ValueError(
                "The number of observations and ground_truths should be"
                "the equal."
            )
        if observations.shape[0] <= 1:
            raise ValueError("At least two observations are needed.")

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

        observation_columns = observations.shape[1]
        coefficients = self.parameters["coefficients"]

        # The number of coefficients should match the number of observation
        if len(coefficients) != observation_columns:
            raise ValueError(
                f"The number of observation columns ({observation_columns}) "
                f"must match the number of coefficients ({len(coefficients)})."
            )
        return self._model.predict(observations)
