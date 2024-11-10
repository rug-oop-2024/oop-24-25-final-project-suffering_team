import os
from abc import ABC, abstractmethod
from glob import glob
from typing import List


class NotFoundError(Exception):
    """The class used to raise a custom error."""

    def __init__(self, path: str) -> None:
        """Initialize the error.

        Args:
            path (str): The path towards a certain file.
        """
        super().__init__(f"Path not found: {path}")


class Storage(ABC):
    """An abstract base class that functions as blueprint for other classes."""

    @abstractmethod
    def save(self, data: bytes, path: str) -> None:
        """Save data to a given path.

        Args:
            data (bytes): Data to save
            path (str): Path to save data
        """
        pass

    @abstractmethod
    def load(self, path: str) -> bytes:
        """Load data from a given path.

        Args:
            path (str): Path to load data
        Returns:
            bytes: Loaded data
        """
        pass

    @abstractmethod
    def delete(self, path: str) -> None:
        """Delete data at a given path.

        Args:
            path (str): Path to delete data
        """
        pass

    @abstractmethod
    def list(self, path: str) -> list:
        """List all paths under a given path.

        Args:
            path (str): Path to list
        Returns:
            list: List of paths
        """
        pass


class LocalStorage(Storage):
    """Class for handling data storage."""

    def __init__(self, base_path: str = "./assets") -> None:
        """Initialize the local storage in the given base_path.

        Args:
            base_path (str, optional): The location where to store the data.
                Defaults to "./assets".
        """
        self._base_path = os.path.normpath(base_path)
        if not os.path.exists(self._base_path):
            os.makedirs(self._base_path)

    def save(self, data: bytes, key: str) -> None:
        """Save the data in the directory with key as filename.

        Args:
            data (bytes): The data to be stored in bytes.
            key (str): The name of file where it should be stored.
        """
        path = self._join_path(key)
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)

    def load(self, key: str) -> bytes:
        """Load the data in the directory with key as filename.

        Args:
            key (str): The name of the file where the data is stored.

        Returns:
            bytes: The data that was stored in the file.
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        with open(path, "rb") as f:
            return f.read()

    def delete(self, key: str = "/") -> None:
        """Delete the file with key as filename.

        Args:
            key (str, optional): The name of the file. Defaults to "/".
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        os.remove(path)

    def list(self, prefix: str = "/") -> List[str]:
        """List all files in the given directory.

        Args:
            prefix (str): The directory where files can be stored.

        Returns:
            List[str]: A list of valid filenames in the directory.
        """
        path = self._join_path(prefix)
        self._assert_path_exists(path)
        keys = glob(os.path.join(path, "**", "*"), recursive=True)
        return [
            os.path.relpath(p, self._base_path)
            for p in keys
            if os.path.isfile(p)
        ]

    def _assert_path_exists(self, path: str) -> None:
        """Check if the path exists.

        Args:
            path (str): The path to be checked.

        Raises:
            NotFoundError: If the path does not exist.
        """
        if not os.path.exists(path):
            raise NotFoundError(path)

    def _join_path(self, path: str) -> str:
        """Create a path from the base_path towards the new file.

        Args:
            path (str): The name of the file to be added to the directory.

        Returns:
            str: The path towards the new file in the directory.
        """
        return os.path.normpath(os.path.join(self._base_path, path))
