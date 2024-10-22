import base64

from pydantic import BaseModel, Field


class Artifact(BaseModel):
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
            type (str): artifact type
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

    def read(self) -> bytes:
        """Retrieve the data from the artifact."""
        return self.data

    def save(self, new_data) -> None:
        """Save the data in the artifact."""
        self.data = new_data
