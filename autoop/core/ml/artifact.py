from pydantic import BaseModel, Field
import base64

class Artifact(BaseModel):
    """
    TO IMPLEMENT:
        save
        read
    """
    def __init__(
        self, 
        name: str, 
        asset_path: str, 
        type: str, 
        data: bytes, 
        version: str, 
        tags: list=[], 
        metadata: dict={}
    ):
        self.name = name 
        self.asset_path = asset_path
        self.type = type
        self.metadata = metadata
        self.data = data
        self.tags = tags
        self.version = version

    def read(self) -> bytes:
        """retrieve the data from the artifact."""
        return self.data
    
    def save(self, new_data) -> None:
        """Save the data in the artifact."""
        self.data = new_data
