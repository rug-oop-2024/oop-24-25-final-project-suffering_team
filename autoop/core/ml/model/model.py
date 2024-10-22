
from abc import abstractmethod, ABC
from autoop.core.ml.artifact import Artifact
import numpy as np
from copy import deepcopy
from typing import Literal

class Model(ABC):
    """Use as an abstract base class for different ML models."""

    def __init__(self):
        self._prameters = {}

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
                A dictionary that replaces the parameter dictionary.

        Raises:
            ValueError:
                Can only replace the dictionary with a dictionary.
            ValueError:
                The keys in the dictionary must be strings.
            ValueError:
                The values in the dictionary must be np.ndarrays.
        """
        if not isinstance(new_parameters, dict):
            raise TypeError("Parameters must be a dictionary.")

        # Check if the dictionary has the correct key and value types
        for key in new_parameters:
            if not isinstance(key, str):
                raise TypeError("Keys for the new_parameters must be strings.")
            if not isinstance(new_parameters[key], np.ndarray):
                raise TypeError(
                    "The values for new_parameters must be np.ndarrays."
                )

        self._parameters = new_parameters

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
