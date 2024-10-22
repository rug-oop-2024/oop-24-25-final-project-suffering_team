from abc import ABC, abstractmethod
from typing import Any

import numpy as np

METRICS = [
    "mean_squared_error",
    "accuracy",
    "mean_absolute_error",
    "r_squared",
]  # add the names (in strings) of the metrics you implement


def get_metric(name: str) -> Any:
    """Create metric using given name.

    Args:
        name (str): name of metric

    Returns:
        Any: instance of metric of given name
    """
    # Factory function to get a metric by name.
    # Return a metric instance given its str name.
    if name not in METRICS:
        return None
    match name:
        case "mean_squared_error":
            return MeanSquaredError()
        case "accuracy":
            return Accuracy()
    return None


class Metric(ABC):
    """Base class for all metrics."""

    def __call__(self):
        """Call get metric function to get the corresponding metric instance."""
        print("what the fuck are ou doing.")

    @abstractmethod
    def evaluate(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """Evaluate the model based on the given metric.

        Args:
            predictions (np.ndarray): an array of predictions
            ground_truth (np.ndarray): an array with the ground_truth
        """
        pass


# add here concrete implementations of the Metric class
class MeanSquaredError(Metric):
    """Create a metric class for mean squared error in regression."""

    def evaluate(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """Evaluate the model's predictions and return the mean squared error.

        Measures how far the predictions are from the mean.

        Args:
            predictions (np.ndarray): An array of predictions
            ground_truth (np.ndarray): An array with the ground_truth

        Returns:
            (float): The mean squared error of the model
        """
        total_squared_error = np.sum((ground_truth - predictions) ** 2)
        return total_squared_error / len(predictions)


class Accuracy(Metric):
    """Create a metric Class for accuracy in classification."""

    def evaluate(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """Evaluate the model's predictions and return the accuracy.

        Measures the ratio of correct predictions.

        Args:
            predictions (np.ndarray): An array of predictions
            ground_truth (np.ndarray): An array with the ground_truth

        Returns:
            (float): The accuracy of the model between 0 and 1
        """
        total_accuracy = 0
        for index in range(len(predictions)):
            value = int(predictions[index] == ground_truth[index])
            total_accuracy += value
        return total_accuracy / len(predictions)


class MeanAbsoluteError(Metric):
    """Create a metric Class for mean absolute error in regression."""

    def evaluate(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """Evaluate the model's predictions and return the mean absolute error.

        Measures the average size of mistakes in a collection of predictions.

        Args:
            predictions (np.ndarray): An array of predictions
            ground_truth (np.ndarray): An array with the ground_truth

        Returns:
            (float): The mean absolute error of the model.
        """
        total_absolute_error = 0
        for index in range(len(predictions)):
            value = abs(ground_truth[index] - predictions[index])
            total_absolute_error += value
        return total_absolute_error / len(predictions)


class RSquared(Metric):
    """Create a metric Class for Rsquared in regression."""

    def evaluate(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """Evaluate the model's predictions and return the rsquared value.

        Measures the proportion of variance that can be explained by the independent
        variables.

        Args:
            predictions (np.ndarray): An array of predictions
            ground_truth (np.ndarray): An array with the ground_truth

        Returns:
            (float): The proportion of variance that can be explained by the independent
            variables of the model between 0 and 1.
        """
        sum_of_squares_regression = np.sum((ground_truth - predictions) ** 2)
        sum_of_squares_total = np.sum((ground_truth - np.mean(ground_truth)) ** 2)
        return 1 - sum_of_squares_regression / sum_of_squares_total
