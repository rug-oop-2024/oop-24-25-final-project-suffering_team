from abc import ABC, abstractmethod
from typing import Any
import numpy as np

METRICS = [
    "mean_squared_error",
    "accuracy",
] # add the names (in strings) of the metrics you implement

def get_metric(name: str):
    # Factory function to get a metric by name.
    # Return a metric instance given its str name.
    if name not in METRICS:
        return None
    match name:
        case "mean_squared_error":
            metric = MeanSquaredError()
        case "accuracy":
            metric = Accuracy()
    return metric

class Metric(ABC):
    """Base class for all metrics.
    """
    # your code here
    # remember: metrics take ground truth and prediction as input and return a real number

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

        mean_squared_error = total_squared_error / len(predictions)
        return mean_squared_error

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
        accuracy = total_accuracy / len(predictions)
        return accuracy

class MeanAbsolutError(Metric):
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
            value = abs(ground_truth[index]-predictions[index])
            total_absolute_error += value
        mean_absolute_error = total_absolute_error / len(predictions)
        return mean_absolute_error

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
        # sum of squares due to regression
        SSR = np.sum((ground_truth - predictions) ** 2)
        # sum of squares total
        SST = np.sum((ground_truth - np.mean(ground_truth)) ** 2)
        r_squared = 1 - SSR / SST
        return r_squared

