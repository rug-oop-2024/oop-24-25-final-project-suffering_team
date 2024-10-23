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
        """Call get metric function to get the correct metric instance."""
        print("what the fuck are ou doing.")

    @abstractmethod
    def evaluate(
        self, predictions: np.ndarray, ground_truth: np.ndarray
    ) -> float:
        """Evaluate the model based on the given metric.

        Args:
            predictions (np.ndarray): an array of predictions
            ground_truth (np.ndarray): an array with the ground_truth
                (Must match number of predictions.)

        Returns:
            (float): The value of the evaluated metric.
        """
        pass

    def _check_dimensions(
        self, predictions: np.ndarray, ground_truth: np.ndarray
    ) -> None:
        """Check if the predictions and ground_truth have the right dimensions.

        Args:
            predictions (np.ndarray): an array of predictions
            ground_truth (np.ndarray): an array with the ground_truth
                (Must match number of predictions.)
        """
        if len(predictions) != len(ground_truth):
            raise ValueError(
                "The number of predictions must equal the number "
                "of ground truth labels."
            )
        if len(predictions) == 0:
            raise ValueError(
                "Predictions and ground truth arrays cannot be empty."
            )


# add here concrete implementations of the Metric class
class MeanSquaredError(Metric):
    """Create a metric class for mean squared error in regression."""

    def evaluate(
        self, predictions: np.ndarray, ground_truth: np.ndarray
    ) -> float:
        """Evaluate the model's mean squared error.

        Measures how far the predictions are from the mean.

        Args:
            predictions (np.ndarray): An array of predictions
            ground_truth (np.ndarray): An array with the ground_truth
                (Must match number of predictions)

        Returns:
            (float): The mean squared error of the model
        """
        self._check_dimensions(predictions, ground_truth)
        total_squared_error = np.sum((ground_truth - predictions) ** 2)
        return total_squared_error / len(predictions)


class MeanAbsoluteError(Metric):
    """Create a metric Class for mean absolute error in regression."""

    def evaluate(
        self, predictions: np.ndarray, ground_truth: np.ndarray
    ) -> float:
        """Evaluate the model's mean absolute error.

        Measures the average size of mistakes in a collection of predictions.

        Args:
            predictions (np.ndarray): An array of predictions
            ground_truth (np.ndarray): An array with the ground_truth
                (Must match number of predictions)

        Returns:
            (float): The mean absolute error of the model.
        """
        self._check_dimensions(predictions, ground_truth)
        total_absolute_error = 0
        for index in range(len(predictions)):
            value = abs(ground_truth[index] - predictions[index])
            total_absolute_error += value
        return total_absolute_error / len(predictions)


class RSquared(Metric):
    """Create a metric Class for Rsquared in regression."""

    def evaluate(
        self, predictions: np.ndarray, ground_truth: np.ndarray
    ) -> float:
        """Evaluate the model's rsquared value.

        Measures the proportion of variance that can be explained
        by the independent variables.

        Args:
            predictions (np.ndarray): An array of predictions
            ground_truth (np.ndarray): An array with the ground_truth
                (Must match number of predictions)

        Returns:
            (float): The proportion of variance that can be explained
                by the independent variables of the model between 0 and 1.
        """
        self._check_dimensions(predictions, ground_truth)
        sum_of_squares_regression = np.sum((ground_truth - predictions) ** 2)
        sum_of_squares_total = np.sum(
            (ground_truth - np.mean(ground_truth)) ** 2
        )
        return 1 - sum_of_squares_regression / sum_of_squares_total


class Accuracy(Metric):
    """Create a metric Class for accuracy in classification."""

    def evaluate(
        self, predictions: np.ndarray, ground_truth: np.ndarray
    ) -> float:
        """Evaluate the model's accuracy.

        Measures the ratio of correct predictions.
        Number of correct predictions / Total number of predictions.

        Args:
            predictions (np.ndarray): An array of prediction labels
            ground_truth (np.ndarray): An array with the ground_truth labels.
                (Must match number of predictions)

        Returns:
            (float): The accuracy of the model between 0 and 1
        """
        self._check_dimensions(predictions, ground_truth)
        if len(predictions) == 0:
            return 0.0

        correct_predictions = np.sum(predictions == ground_truth)
        return correct_predictions / len(predictions)


class Precision(Metric):
    """Create a metric Class for precision in classification."""

    def evaluate(
        self, predictions: np.ndarray, ground_truth: np.ndarray
    ) -> float:
        """Evaluate the model's precision.

        Precision = True positive / (True positive + False positive)

        Args:
            predictions (np.ndarray): An array of prediction labels
            ground_truth (np.ndarray): An array with the ground_truth labels
                (Must match number of predictions)

        Returns:
            (float): The precision of the model between 0 and 1
        """
        self._check_dimensions(predictions, ground_truth)
        unique_labels = np.unique(ground_truth)
        num_unique_labels = len(unique_labels)

        if num_unique_labels == 0:
            return 0.0

        total_precision = 0.0

        for unique_label in np.unique(ground_truth):
            total_precision += self._calculate_label_precision(
                unique_label, predictions, ground_truth
            )

        return total_precision / num_unique_labels

    def _calculate_label_precision(
        self,
        unique_label: int | str,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
    ) -> float:
        """Evaluate the model's precision of one label.

        Precision = True positive / (True positive + False positive)

        Args:
            unique_label (int | str): The label for which the models precision
                needs to be calculated.
            predictions (np.ndarray): An array of prediction labels
            ground_truth (np.ndarray): An array with the ground_truth labels
                (Must match number of predictions)

        Returns:
            (float): The precision of the model between 0 and 1 of one label.
        """
        # Create boolean arrays that indicate matches for the unique label
        # in both predictions and ground truth.
        match_gt = ground_truth == unique_label
        match_pred = predictions == unique_label

        # Count the true positives and false positives using the arrays.
        true_pos = np.sum(match_gt & match_pred)
        false_pos = np.sum(~match_gt & match_pred)

        # Avoid dividing by zero
        if true_pos + false_pos > 0:
            return true_pos / (true_pos + false_pos)

        return 0.0
