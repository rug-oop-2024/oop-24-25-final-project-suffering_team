import numpy as np

from autoop.core.ml.model.model import Model


class MultipleLinearRegression(Model):
    """A MultipleLinearRegression implementation of the model class."""

    def fit(self, observations: np.ndarray, ground_truths: np.ndarray) -> None:
        """Train the model using observations and ground_truths.

        This method calculates the optimal parameter configuration to describe
        the relation between observations and the corresponding ground_truths.

        Args:
            observations (np.ndarray):
                Observations used to train the model. Row dimension is
                samples, column dimension is variables.
            ground_truths (np.ndarray):
                Ground_truths corresponding to the observations used to train
                the model. Row dimension is samples.

        Raises:
            ValueError:
                The number of observations and ground_truths must be equal.
            ValueError:
                At least two observations are needed for regression.
        """
        # Add a column of ones for the intercept
        n_rows = observations.shape[0]
        column_of_ones = np.ones((n_rows, 1))
        np.append(observations, column_of_ones, axis=1)
        # Check if there rows are even
        observation_rows = observations.shape[0]
        ground_truth_rows = ground_truths.shape[0]
        if observation_rows != ground_truth_rows:
            raise ValueError(
                "The number of observations and ground_truths should be"
                "the equal."
            )

        # Check if there are at least two observations
        if observation_rows <= 1:
            raise ValueError("At least two observations are needed.")

        # Calculate the parameters using z_star = (X^T * X)^-1 * X^T * y
        transposed_observations = np.transpose(observations)
        result = np.matmul(transposed_observations, observations)
        result = np.linalg.inv(result)
        result = np.matmul(result, transposed_observations)
        result = np.matmul(result, ground_truths)
        self.parameters = {"parameters": result}

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Return a prediction for the observations in the current model.

        Uses the internal parameters to predict what the values should be of
        given observations.

        Args:
            observations (np.ndarray):
                The observation for which a prediction is asked. Row
                dimension is samples, column dimension is variables.

        Raises:
            ValueError:
                There are no parameters, the model needs to be fitted first.
            ValueError:
                For predictions, the number of observation columns should
                equal the number of parameter rows.

        Returns:
            np.ndarray:
                The predictions for the given observations.
        """
        # Add a column of ones for the intercept
        n_rows = observations.shape[0]
        column_of_ones = np.ones((n_rows, 1))
        np.append(observations, column_of_ones, axis=1)

        # Check for and store the parameters
        params = self.parameters
        if "parameters" not in params.keys():
            raise ValueError(
                "Model not fitted. Call 'fit' with appropriate arguments"
                "before using 'predict'"
            )

        parameter_matrix = params["parameters"]

        # Parameter rows must equal observation columns
        observation_columns = observations.shape[1]
        parameter_rows = len(parameter_matrix)
        if parameter_rows != observation_columns:
            raise ValueError(
                f"The number of observation columns ({observation_columns}) "
                f"is not the same as the number of parameter rows"
                f"({parameter_rows})."
            )

        # Calculate the prediction using y = X*w_star
        return np.matmul(observations, parameter_matrix)
