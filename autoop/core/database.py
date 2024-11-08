import json
import os
from typing import List, Optional, Tuple

from autoop.core.storage import Storage


class Database:
    """A class for database."""

    def __init__(self, storage: Storage) -> None:
        """Initialize the database.

        Args:
            storage (Storage): _description_
        """
        self._storage = storage
        self._data = {}
        self._load()

    def set_data(self, collection: str, data_id: str, entry: dict) -> dict:
        """Set a key in the database.

        Args:
            collection (str): The collection to store the data in
            data_id (str): The id of the data
            entry (dict): The data to store
        Returns:
            dict: The data that was stored
        """
        assert isinstance(entry, dict), "Data must be a dictionary"
        assert isinstance(collection, str), "Collection must be a string"
        assert isinstance(data_id, str), "ID must be a string"
        if not self._data.get(collection):
            self._data[collection] = {}
        self._data[collection][data_id] = entry
        self._persist()
        return entry

    def get(self, collection: str, data_id: str) -> Optional[dict]:
        """Get a key from the database.

        Args:
            collection (str): The collection to get the data from
            data_id (str): The id of the data
        Returns:
            Optional[dict]: The data that was stored,
                or None if it doesn't exist
        """
        if not self._data.get(collection):
            return None
        return self._data[collection].get(data_id)

    def delete(self, collection: str, data_id: str) -> None:
        """Delete a key from the database.

        Args:
            collection (str): The collection to delete the data from
            data_id (str): The id of the data
        Returns:
            None
        """
        if not self._data.get(collection):
            return
        if self._data[collection].get(data_id):
            del self._data[collection][data_id]
        self._persist()

    def data_list(self, collection: str) -> List[Tuple[str, dict]]:
        """List all data in a collection.

        Args:
            collection (str): The collection to list the data from
        Returns:
            List[Tuple[str, dict]]: A list of tuples containing the id and
                data for each item in the collection
        """
        if not self._data.get(collection):
            return []
        return [
            (data_id, data) for data_id, data in self._data[collection].items()
        ]

    def refresh(self) -> None:
        """Refresh the database by loading the data from storage."""
        self._load()

    def _persist(self) -> None:
        """Persist the data to storage."""
        for collection, data in self._data.items():
            if not data:
                continue
            for data_id, item in data.items():
                self._storage.save(
                    json.dumps(item).encode(), f"{collection}{os.sep}{data_id}"
                )

        # for things that were deleted, we need to remove them from the storage
        keys = self._storage.list("")
        for key in keys:
            collection, data_id = key.split(os.sep)[-2:]
            if not self._data[collection].get(data_id):
                self._storage.delete(f"{collection}{os.sep}{data_id}")

    def _load(self) -> None:
        """Load the data from storage."""
        self._data = {}
        for key in self._storage.list(""):
            collection, data_id = key.split(os.sep)[-2:]
            data = self._storage.load(f"{collection}{os.sep}{data_id}")
            # Ensure the collection exists in the dictionary
            if collection not in self._data:
                self._data[collection] = {}
            self._data[collection][data_id] = json.loads(data.decode())
