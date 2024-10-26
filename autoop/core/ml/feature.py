class Feature:
    """Represent a categorical or numerical column in a csv."""

    def __init__(self, name: str, column_type: str):
        """Initialize the feature.

        Args:
            name (str): The name of the feature.
            column_type (str): The type of feature, categorical or numerical.
        """
        self.name = name
        self.type = column_type

    def __str__(self) -> str:
        """Return the name and variables of the feature.

        Returns:
            str: The string representation of the name and variables of the
                feature.
        """
        return f"Column is {self.name}, which contains {self.type} variables."
