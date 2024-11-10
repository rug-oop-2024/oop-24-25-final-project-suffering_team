import base64
from copy import deepcopy


class Artifact:
    """Make artifact class."""

    def __init__(
        self,
        name: str,
        data: bytes,
        artifact_type: str = None,
        asset_path: str = None,
        version: str = None,
        tags: list = None,
        metadata: dict = None,
    ) -> None:
        """Initialize artifact.

        Args:
            name (str): name of artifact
            asset_path (str, opotional): path of artifact. Default to name.
            artifact_type (str, optional): artifact type. Default to other.
            data (bytes): data of artifact
            version (str, optional): artifact version. Default to 1.0.0.
            tags (list, optional): tags for artifact. Defaults to [].
            metadata (dict, optional): ids of metadata. Defaults to {}.
        """
        self._name = name
        self._asset_path = asset_path if asset_path is not None else name
        self._type = artifact_type if artifact_type is not None else "other"
        self._data = data
        self._version = version if version is not None else "1.0.0"
        self._tags = tags if tags is not None else []
        self._metadata = metadata if metadata is not None else {}
        self._id = f"{self._base64_encode(self.asset_path)}-{self.version}"

    @property
    def name(self) -> str:
        """Get name of artifact.

        Returns:
            str: name of artifact
        """
        return self._name

    @property
    def asset_path(self) -> str:
        """Get asset_path of artifact.

        Returns:
            str: asset_path of artifact
        """
        return self._asset_path

    @property
    def type(self) -> str:
        """Get type of artifact.

        Returns:
            str: type of artifact
        """
        return self._type

    @property
    def data(self) -> bytes:
        """Get data of artifact.

        Returns:
            bytes: data of artifact
        """
        return self._data

    @property
    def version(self) -> str:
        """Get version of artifact.

        Returns:
            str: version of artifact
        """
        return self._version

    @property
    def tags(self) -> list:
        """Get tags of artifact.

        Returns:
            list: tags of artifact
        """
        return deepcopy(self._tags)

    @property
    def metadata(self) -> dict:
        """Get metadata of artifact.

        Returns:
            dict: metadata of artifact
        """
        return deepcopy(self._metadata)

    @property
    def id(self) -> str:
        """Get id of artifact.

        Returns:
            str: id of artifact
        """
        return self._id

    def save_metadata(self, artifact: "Artifact") -> None:
        """Save new metadata.

        Args:
            artifact (Artifact): metadata to be saved
        """
        self._metadata.update({artifact.name: artifact.id})

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

    def save(self, new_data: bytes) -> None:
        """Save the data in the artifact.

        Args:
            new_data (bytes): The data that is to be saved in the artifact.

        Returns:
            returns the data that was given.
        """
        self._data = new_data
        return self.data
