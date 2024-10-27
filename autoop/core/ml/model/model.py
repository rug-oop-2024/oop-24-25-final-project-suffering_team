from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any

import numpy as np


class Model(ABC):
    """Use as an abstract base class for different ML models."""

    def __init__(self):
        """Initialize model base class."""
        self._parameters = {}
        self._type = "Type has not been set."  # Set type in subclass models

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
    def parameters(self, new_parameters: dict[str, Any]) -> None:
        """Use the setter to validate and update the model parameters.

        Args:
            new_parameters (dict[str, np.ndarray]):
                A dictionary that updates the parameter dictionary.

        Raises:
            ValueError:
                If the new input is not a dictionary.
        """
        if not isinstance(new_parameters, dict):
            raise TypeError("Parameters must be a dictionary.")

        # Check if the dictionary has the correct key and value types
        for key in new_parameters.keys():
            self._validate_key(key)

        self._parameters.update(new_parameters)

    @property
    def type(self) -> str:
        """Get the model type."""
        return self._type

    @type.setter
    def type(self, model_type: str) -> None:
        """Set the model type after checking that it is a string.

        Args:
            model_type (str): The type of the model, must be a string.

        Raises:
            TypeError: If model_type is not a string.
        """
        if not isinstance(model_type, str):
            raise TypeError("Model type must be a string.")
        self._type = model_type

    def _validate_key(self, key: str) -> None:
        """Validate individual keys for the parameters dictionary.

        Args:
            key (str):
                The parameter name, which must be a string.

        Raises:
            TypeError:
                If the key is not a string.
        """
        if not isinstance(key, str):
            raise TypeError("Keys for the parameters must be strings.")

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
