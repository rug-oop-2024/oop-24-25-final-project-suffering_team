
from pydantic import BaseModel, Field
from typing import Literal
import numpy as np

from autoop.core.ml.dataset import Dataset

class Feature(BaseModel):
    """Represent a categorical or numerical column in a csv."""
    def __init__(self, name: str, column_type: str):
        """Initialize the feature.
        
        Args:
            name (str): The name of the feature.
            column_type (str): The type of feature, categorical or numerical.
        """
        self.name = name
        self.type = column_type

    def __str__(self):
        given_string = f"Column to predict is {self.name}, which contains {self.type} variables."
        return given_string

    