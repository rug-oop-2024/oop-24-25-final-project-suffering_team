from abc import ABC, abstractmethod

import numpy as np

METRICS = [
    "mean_squared_error",
    "mean_absolute_error",
    "r_squared",
    "accuracy",
    "precision",
    "recall",
]


def get_metric(name: str) -> "Metric":
    """Return a metric instance corresponding to the provided name.

    Args:
        name (str): The name of the asked metric.

    Returns:
        Metric: An instance of the specified metric.

    Raises:
        ValueError: If the metric name is not valid.
    """
    if name not in METRICS:
        raise ValueError(
            f"'{name}' is not a valid metric name. Use one of the following:\n"
            + "\n".join(METRICS)
        )

    match name:
        case "mean_squared_error":
            return MeanSquaredError()
        case "mean_absolute_error":
            return MeanAbsoluteError()
        case "r_squared":
            return RSquared()
        case "accuracy":
            return Accuracy()
        case "precision":
            return Precision()
        case "recall":
            return Recall()

    raise ValueError(f"No metric found for the name: '{name}'.")


class Metric(ABC):
    """Base class for all metrics."""

    def __call__(
        self, predictions: np.ndarray, ground_truth: np.ndarray
    ) -> float:
        """Evaluate predictions with the class-defined metric.

        Args:
            predictions (np.ndarray): An array of predictions.
            ground_truth (np.ndarray): An array with the ground_truth
                (Must match number of predictions.)

        Returns:
            float: The value of the evaluated metric.
        """
        return self.evaluate(predictions, ground_truth)

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
            float: The value of the evaluated metric.
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

        Raises:
            ValueError: If the number of predictions does not equal the number
                of ground truth labels.
            ValueError: If there are no predictions or ground_truths.
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
            float: The mean squared error of the model
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
            float: The mean absolute error of the model.
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
            float: The proportion of variance that can be explained
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
            float: The accuracy of the model between 0 and 1
        """
        self._check_dimensions(predictions, ground_truth)

        correct_predictions = np.sum(predictions == ground_truth)
        return correct_predictions / len(predictions)


class Precision(Metric):
    """Create a metric Class for precision in classification."""

    def evaluate(
        self, predictions: np.ndarray, ground_truth: np.ndarray
    ) -> float:
        """Evaluate the model's precision.

        Measures the accuracy of positive predictions for each label.
        Precision = True positive / (True positive + False positive)

        Args:
            predictions (np.ndarray): An array of prediction labels
            ground_truth (np.ndarray): An array with the ground_truth labels
                (Must match number of predictions)

        Returns:
            float: The precision of the model between 0 and 1
        """
        self._check_dimensions(predictions, ground_truth)

        unique_labels = np.unique(ground_truth)
        num_unique_labels = len(unique_labels)

        total_precision = 0.0

        for unique_label in unique_labels:
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
            float: The precision of the model between 0 and 1 of one label.
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


class Recall(Metric):
    """Create a metric Class for recall in classification."""

    def evaluate(
        self, predictions: np.ndarray, ground_truth: np.ndarray
    ) -> float:
        """Evaluate the model's recall ability.

        Measures the classifier's ability to find all correct predictions for
        each label. Recall = True positive / (True positive + False negative)

        Args:
            predictions (np.ndarray): An array of prediction labels
            ground_truth (np.ndarray): An array with the ground_truth labels
                (Must match number of predictions)

        Returns:
            float: The recall of the model between 0 and 1
        """
        self._check_dimensions(predictions, ground_truth)

        unique_labels = np.unique(ground_truth)
        num_unique_labels = len(unique_labels)

        total_recall = 0.0

        for unique_label in unique_labels:
            total_recall += self._calculate_label_recall(
                unique_label, predictions, ground_truth
            )

        return total_recall / num_unique_labels

    def _calculate_label_recall(
        self,
        unique_label: int | str,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
    ) -> float:
        """Evaluate the model's recall of one label.

        Recall = True positive / (True positive + False negative)

        Args:
            unique_label (int | str): The label for which the models recall
                needs to be calculated.
            predictions (np.ndarray): An array of prediction labels
            ground_truth (np.ndarray): An array with the ground_truth labels
                (Must match number of predictions)

        Returns:
            float: The recall of the model between 0 and 1 of one label.
        """
        # Create boolean arrays that indicate matches for the unique label
        # in both predictions and ground truth.
        match_gt = ground_truth == unique_label
        match_pred = predictions == unique_label

        # Count the true positives and false negatives using the arrays.
        true_pos = np.sum(match_gt & match_pred)
        false_neg = np.sum(match_gt & ~match_pred)

        # Avoid dividing by zero
        if true_pos + false_neg > 0:
            return true_pos / (true_pos + false_neg)

        return 0.0
