from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Literal

import numpy as np

from autoop.core.ml.artifact import Artifact


class Model(ABC):
    """Use as an abstract base class for different ML models."""

    def __init__(self):
        self._parameters = {}

    @property
    def parameters(self) -> dict[str, np.ndarray]:
        """Get the model's parameters dictionary.

        Returns:
            dict[str, np.ndarray]:
                A deepcopy of the dictionary containing the parameters used
                by the model.
        """
        # returns a deepcopy to avoid leakage
        return deepcopy(self._parameters)

    @parameters.setter
    def parameters(self, new_parameters: dict[str, np.ndarray]) -> None:
        """Use the setter to validate and update the model parameters.

        Args:
            value (dict[str, np.ndarray]):
                A dictionary that updates the parameter dictionary.

        Raises:
            ValueError:
                If the new input is not a dictionary.
        """
        if not isinstance(new_parameters, dict):
            raise TypeError("Parameters must be a dictionary.")

        # Check if the dictionary has the correct key and value types
        for key, value in new_parameters.items:
            self._validate_key_value(key, value)

        self._parameters.update(new_parameters)

    def _validate_key_value(self, key: str, value: np.ndarray) -> None:
        """Validate individual key-value pairs for the parameters dictionary.

        Args:
            key (str):
                The parameter name, which must be a string.
            value (np.ndarray):
                The parameter value, which must be a numpy ndarray.

        Raises:
            TypeError:
                If the key is not a string or the value is not a numpy ndarray.
        """
        if not isinstance(key, str):
            raise TypeError("Keys for the parameters must be strings.")
        if not isinstance(value, np.ndarray):
            raise TypeError("Values for parameters must be np.ndarrays.")

    @abstractmethod
    def fit(self, observations: np.ndarray, ground_truths: np.ndarray) -> None:
        """Train the model using observations and ground_truths.

        Args:
            observations (np.ndarray):
                Observations used to train the model. Row dimension is
                samples, column dimension is variables.
            ground_truths (np.ndarray):
                Ground_truths corresponding to the observations used to
                train the model. Row dimension is samples.
        """
        pass

    @abstractmethod
    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Make predictions for observations using the model.

        Args:
            observations (np.ndarray):
                The observations which need predictions. Row dimension is
                samples, column dimensionis variables.

        Returns:
            np.ndarray:
                The predictions for the observations.
        """
        pass
