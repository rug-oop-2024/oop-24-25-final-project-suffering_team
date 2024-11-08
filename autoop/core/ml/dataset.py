import io

from autoop.core.ml.artifact import Artifact

import pandas as pd


class Dataset(Artifact):
    """A class for datasets."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the dataset."""
        super().__init__(artifact_type="dataset", *args, **kwargs)

    @staticmethod
    def from_dataframe(
        data: pd.DataFrame, name: str, asset_path: str, version: str = "1.0.0"
    ) -> "Dataset":
        """Convert a dataframe into a dataset with a name and asset path.

        Args:
            data (pd.DataFrame): The data as pandas dataframe.
            name (str): The name of the dataset.
            asset_path (str): The path to where the artifact is stored.
            version (str, optional): The artifact version. Defaults to "1.0.0".

        Returns:
            Dataset: The dataset recovered from the dataframe.
        """
        return Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
        )

    def read(self) -> pd.DataFrame:
        """Create a csv file from the stored data.

        Returns:
            pd.DataFrame: The data frame restored from the bytes.
        """
        data_bytes = super().read()
        csv = data_bytes.decode()
        return pd.read_csv(io.StringIO(csv))

    def save(self, data: pd.DataFrame) -> bytes:
        """Convert the dataframe into bytes and store it in the artifact.

        Args:
            data (pd.DataFrame): The data that is to be converted into bytes.

        Returns:
            bytes: The bytes representing the dataframe.
        """
        data_bytes = data.to_csv(index=False).encode()
        return super().save(data_bytes)
