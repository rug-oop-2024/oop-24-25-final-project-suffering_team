class DatasetValidationError(Exception):
    """Exception for missing features or incorrect types in a dataset."""

    def __init__(
        self,
        message: str = "Dataset validation failed:",
        missing_features: list[str] = None,
        incorrect_types: dict[str, tuple[str, str]] = None,
        extra_features: list[str] = None,
    ) -> None:
        """Initialize the error message.

        Args:
            message (str, optional): The message that preceeds the error
                details. Defaults to "Dataset validation failed:".
            missing_features (list[str], optional): The necesessary features
                which are missing from the current dataset. Defaults to None.
            incorrect_types (dict[str, tuple[str, str]], optional): The
                features which have the wrong type. Defaults to None.
                dict(feature_name: (given_type, expected_type))
            extra_features (list[str], optional): Extra features which should
                not be included in the dataset. Defaults to None.
        """
        self._message = message
        self._missing_features = missing_features or []
        self._incorrect_types = incorrect_types or {}
        self._extra_features = extra_features or []
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Add details about the error to the message.

        Returns:
            str: The message with added details about the error.
        """
        details = [self._message]
        if self._missing_features:
            details.append(
                f"Missing features: {', '.join(self._missing_features)}."
            )
        if self._incorrect_types:
            details.append("Incorrect types: ")
            type_info = ", ".join(
                f"Feature '{feature}' is of type '{given}' "
                f"while it should be '{expected}'"
                for feature, (given, expected) in self._incorrect_types.items()
            )
            details.append(type_info)
        if self._extra_features:
            details.append(
                f"Extra features: {', '.join(self._extra_features)}."
            )
        details.append("Change the dataset and try again.")
        return " ".join(details)

    def __str__(self) -> str:
        """Return the error in string format.

        Returns:
            str: The error in string format.
        """
        return self._format_message()
