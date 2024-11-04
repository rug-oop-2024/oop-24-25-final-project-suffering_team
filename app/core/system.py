from typing import List

from autoop.core.database import Database
from autoop.core.ml.artifact import Artifact
from autoop.core.storage import LocalStorage, Storage

import streamlit as st


class ArtifactRegistry:
    """Class to register artifacts."""

    def __init__(self, database: Database, storage: Storage):
        """Initialize registry.

        Args:
            database (Database): The database
            storage (Storage): local storage
        """
        self._database = database
        self._storage = storage

    def register(self, artifact: Artifact):
        """Register the artifact.

        Args:
            artifact (Artifact): artifact to register
        """
        # save the artifact in the storage
        self._storage.save(artifact.data, artifact.asset_path)
        # save the metadata in the database
        entry = {
            "name": artifact.name,
            "version": artifact.version,
            "asset_path": artifact.asset_path,
            "tags": artifact.tags,
            "metadata": artifact.metadata,
            "type": artifact.type,
        }
        self._database.set_data(f"artifacts", artifact.id, entry)

    def list(self, type: str = None) -> List[Artifact]:
        """Get artrifacts in the registry

        Args:
            type (str, optional): type of artifact you need. Defaults to None.

        Returns:
            List[Artifact]: list of artifacts
        """
        entries = self._database.data_list("artifacts")
        artifacts = []
        for id, data in entries:
            if type is not None and data["type"] != type:
                continue
            artifact = Artifact(
                name=data["name"],
                version=data["version"],
                asset_path=data["asset_path"],
                tags=data["tags"],
                metadata=data["metadata"],
                data=self._storage.load(data["asset_path"]),
                artifact_type=data["type"],
            )
            artifacts.append(artifact)
        return artifacts

    def get(self, artifact_id: str) -> Artifact:
        """Get certain artifact.

        Args:
            artifact_id (str): id of artifact
        """
        data = self._database.get("artifacts", artifact_id)
        st.write(data)
        return Artifact(
            name=data["name"],
            version=data["version"],
            asset_path=data["asset_path"],
            tags=data["tags"],
            metadata=data["metadata"],
            data=self._storage.load(data["asset_path"]),
            artifact_type=data["type"],
        )

    def delete(self, artifact_id: str):
        """Delete artifact from registry.

        Args:
            artifact_id (str): id of artifact
        """
        data = self._database.get("artifacts", artifact_id)
        self._storage.delete(data["asset_path"])
        self._database.delete("artifacts", artifact_id)


class AutoMLSystem:
    """Class for system."""

    _instance = None

    def __init__(self, storage: LocalStorage, database: Database):
        """Initialize system.

        Args:
            storage (LocalStorage): storage of system
            database (Database): database of objects
        """
        self._storage = storage
        self._database = database
        self._registry = ArtifactRegistry(database, storage)

    @staticmethod
    def get_instance():
        """Get an instance of this class.

        Returns:
            AutoMlSystem: returns instance of this class
        """
        if AutoMLSystem._instance is None:
            AutoMLSystem._instance = AutoMLSystem(
                LocalStorage("./assets/objects"),
                Database(LocalStorage("./assets/dbo")),
            )
        AutoMLSystem._instance._database.refresh()
        return AutoMLSystem._instance

    @property
    def registry(self):
        """Get artifact registry.

        Returns:
            ArtifactRegistry: return current registry
        """
        return self._registry
