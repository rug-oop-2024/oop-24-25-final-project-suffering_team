import base64


class Artifact:
    """Make artifact class."""

    def __init__(
        self,
        name: str,
        asset_path: str,
        artifact_type: str,
        data: bytes,
        version: str,
        tags: list = None,
        metadata: dict = None,
    ):
        """Initialize artifact.

        Args:
            name (str): name of artifact
            asset_path (str): path of artifact
            artifact_type (str): artifact type
            data (bytes): data of artifact
            version (str): artifact version
            tags (list, optional): tags for artifact. Defaults to [].
            metadata (dict, optional): ids of metadata. Defaults to {}.
        """
        self.name = name
        self.asset_path = asset_path
        self.type = artifact_type
        self.data = data
        self.version = version
        self.tags = tags if tags is not None else []
        self.metadata = metadata if metadata is not None else {}
        self.id = f"{self._base64_encode(self.asset_path)}-{self.version}"

    def save_metadata(self, artifact: "Artifact"):
        """Save new metadata.

        Args:
            name (str): name of metadata
        """
        self.metadata.update({artifact.name: artifact.id})

    @staticmethod
    def _base64_encode(value: str) -> str:
        """Encode a string using base64 for unique asset ID generation.

        Args:
            value (str): The value to encode.

        Returns:
            str: Base64 encoded string.
        """
        return base64.urlsafe_b64encode(value.encode()).decode()

    def read(self) -> bytes:
        """Retrieve the data from the artifact.

        Returns:
            bytes: The bytes that represent the data of the artifact.
        """
        return self.data

    def save(self, new_data) -> None:
        """Save the data in the artifact.

        Args:
            new_data (_type_): The data that is to be saved in the artifact.

        Returns:
            returns the data that was given.
        """
        self.data = new_data
        return self.data
