class Feature:
    """Represent a categorical or numerical column in a csv."""

    def __init__(self, name: str, column_type: str) -> None:
        """Initialize the feature.

        Args:
            name (str): The name of the feature.
            column_type (str): The type of feature, categorical or numerical.
        """
        self._name = name
        self._type = column_type

    @property
    def name(self) -> str:
        """Get name of feature.

        Returns:
            str: name of feature
        """
        return self._name

    @property
    def type(self) -> str:
        """Get type of feature.

        Returns:
            str: type of feature
        """
        return self._type

    def __str__(self) -> str:
        """Return the name and variables of the feature.

        Returns:
            str: The string representation of the name and variables of the
                feature.
        """
        return f"Column is {self.name}, which contains {self.type} variables."
