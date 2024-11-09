import io
import pickle
from typing import TYPE_CHECKING, List

from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric

if TYPE_CHECKING:
    from autoop.core.ml.model import Model

from autoop.functional.feature import detect_feature_types
from autoop.functional.preprocessing import preprocess_features

import numpy as np
import pandas as pd

from exceptions import DatasetValidationError


class Pipeline:
    """A pipeline class to bring together different dataprocessing classes."""

    def __init__(
        self,
        metrics: List[Metric],
        dataset: Dataset,
        model: "Model",
        input_features: List[Feature],
        target_feature: Feature,
        split: float = 0.8,
    ) -> None:
        """Initialize the pipeline.

        Args:
            metrics (List[Metric]): The list of metrics to be used.
            dataset (Dataset): The data used for processing.
            model (Model): The model used to process the data.
            input_features (List[Feature]): A list of features stating
                their name and whether they are numerical or categorical.
            target_feature (Feature): The feature to make predictions for.
            split (float, optional): The split of the train and testing data.
                Defaults to 0.8.

        Raises:
            ValueError: If a classification model is used on a continuous
                target feature.
            ValueError: If a regression model is used on a catagorical target
                feature.
        """
        self._dataset = dataset
        self._model = model
        self._input_features = input_features
        self._target_feature = target_feature
        self._metrics = metrics
        self._artifacts = {}
        self._split = split
        if target_feature.type == "categorical":
            if model.type != "classification":
                raise ValueError(
                    "Model type must be classification",
                    "for categorical target feature",
                )
        if target_feature.type == "continuous" and model.type != "regression":
            raise ValueError(
                "Model type must be regression for continuous target feature"
            )

    def __str__(self) -> str:
        """Return a string representation of the most important attributes."""
        return f"""
            Pipeline(
                model={self._model.type},\n
                input_features={list(map(str, self._input_features))},\n
                target_feature={str(self._target_feature)},\n
                split={self._split},\n
                metrics={list(map(str, self._metrics))},
        )
        """

    @property
    def model(self) -> "Model":
        """Return the model used by the current Pipeline."""
        return self._model

    @property
    def artifacts(self) -> List[Artifact]:
        """Save the artifacts generated during pipeline execution."""
        artifacts = []
        for name, artifact in self._artifacts.items():
            artifact_type = artifact.get("type")
            if artifact_type in ["OneHotEncoder"]:
                data = artifact["encoder"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
            if artifact_type in ["StandardScaler"]:
                data = artifact["scaler"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
        pipeline_data = {
            "dataset": self._dataset,
            "input_features": self._input_features,
            "target_feature": self._target_feature,
            "split": self._split,
            "metrics": self._metrics,
        }
        # Store metric results if they exist.
        if hasattr(self, "_train_metrics_results"):
            pipeline_data.update(
                {
                    "train_results": self._train_metrics_results,
                    "test_results": self._metrics_results,
                }
            )

        artifacts.append(
            Artifact(name="pipeline_config", data=pickle.dumps(pipeline_data))
        )
        artifacts.append(
            self.model.to_artifact(name=f"pipeline_model_{self.model.type}")
        )
        return artifacts

    def _register_artifact(self, name: str, artifact: Artifact) -> None:
        """Store the artifact instance by name in the artifact dictionary."""
        self._artifacts[name] = artifact

    def _preprocess_features(self) -> None:
        """Preprocess the input and target features from the dataset.

        1. Preprocess the target feature and register the artifact.
        2. Preprocess the input features and register their artifacts.
        3. Store the output vector and input vectors for later use during
        model training and evaluation.
        """
        target_feature_name, target_data, artifact = preprocess_features(
            [self._target_feature], self._dataset
        )[0]
        self._register_artifact(target_feature_name, artifact)
        input_results = preprocess_features(
            self._input_features, self._dataset
        )
        for feature_name, _data, artifact in input_results:
            self._register_artifact(feature_name, artifact)
        # Get the input vectors and output vector,
        # sort by feature name for consistency
        self._output_vector = target_data
        self._input_vectors = [
            data for (feature_name, data, artifact) in input_results
        ]

    def _split_data(self) -> None:
        """Split the data into training and testing sets."""
        split = self._split
        self._train_X = [
            vector[:int(split * len(vector))]
            for vector in self._input_vectors
        ]
        self._test_X = [
            vector[int(split * len(vector)):]
            for vector in self._input_vectors
        ]
        self._train_y = self._output_vector[
            :int(split * len(self._output_vector))
        ]
        self._test_y = self._output_vector[
            int(split * len(self._output_vector)):
        ]

    def _compact_vectors(self, vectors: List[np.array]) -> np.array:
        """Combine a list of vectors into one array.

        Args:
            vectors (List[np.array]): The list of vectors which need
                to be combined.

        Returns:
            np.array: The combined vectors in one array.
        """
        return np.concatenate(vectors, axis=1)

    def _train(self) -> None:
        """Fit the model using the training data."""
        observations = self._compact_vectors(self._train_X)
        ground_truth = self._train_y
        self._model.fit(observations, ground_truth)

    def _evaluate(self) -> None:
        """Predict values for the data and collect the metric results."""
        observations = self._compact_vectors(self._train_X)
        ground_truth = self._train_y
        self._train_metrics_results = []
        predictions = self._model.predict(observations)
        for metric in self._metrics:
            result = metric.evaluate(predictions, ground_truth)
            self._train_metrics_results.append((metric, result))

        observations = self._compact_vectors(self._test_X)
        ground_truth = self._test_y
        self._metrics_results = []
        predictions = self._model.predict(observations)
        for metric in self._metrics:
            result = metric.evaluate(predictions, ground_truth)
            self._metrics_results.append((metric, result))
        encoder = list(self._artifacts[self._target_feature.name].values())[1]
        if encoder.__class__.__name__ == "StandardScaler":
            # Restore the original values using the encoder
            predictions = encoder.inverse_transform(predictions)
        else:
            # Restore the original labels using the encoder
            predictions = [
                encoder.categories_[0][index] for index in predictions
            ]
        self._predictions = predictions

    def execute(self) -> dict[str, list]:
        """Process the data in the model and collect the results.

        Returns:
            dict[str, list]: A dictionary containing the training and test
                metrics and the predictions.
        """
        self._preprocess_features()
        self._split_data()
        self._train()
        self._evaluate()
        return {
            "train_metrics": self._train_metrics_results,
            "test_metrics": self._metrics_results,
            "predictions": self._predictions,
        }

    def _validate_prediction_features(self, new_dataset: Dataset) -> None:
        """Validate that only trained features are in the dataset.

        Args:
            new_data (Dataset): The new dataset that is to be checked.

        Raises:
            DatasetValidationError: If necessary features are missing.
            DatasetValidationError: If there are extra features.
            DatasetValidationError: If the features are of a wrong type.
        """
        # Collect the feature names and types from the data set
        new_features = {
            feature.name: feature.type
            for feature in detect_feature_types(new_dataset)
        }

        # Store the required feature names and types
        required_features = [feature.name for feature in self._input_features]
        required_types = {
            feature.name: feature.type for feature in self._input_features
        }

        # Check if the required features are present in the new_features
        missing_features = [
            feature
            for feature in required_features
            if feature not in new_features
        ]
        if missing_features:
            raise DatasetValidationError(missing_features=missing_features)

        # Check if there are features which should not be present
        extra_features = [
            feature
            for feature in new_features
            if feature not in required_features
        ]
        if extra_features:
            raise DatasetValidationError(extra_features=extra_features)

        # Check if the present features have the correct type
        incorrect_types = {
            feature: (new_features[feature], expected_type)
            for feature, expected_type in required_types.items()
            if new_features[feature] != expected_type
        }
        if incorrect_types:
            raise DatasetValidationError(incorrect_types=incorrect_types)

    def _preprocess_prediction_columns(self, new_dataset: Dataset) -> None:
        """Reorder the new columns so they are in the expected order.

        Args:
            new_data (Dataset): The dataset with columns that need sorting.
        """
        csv = new_dataset.data.decode()
        full_data = pd.read_csv(io.StringIO(csv))
        expected_column_order = [
            feature.name for feature in self._input_features
        ]
        # Reorder the columns using the list of column names in the correct
        new_data_reordered = full_data[expected_column_order]
        new_dataset.data = new_data_reordered.to_csv(index=False).encode()
        input_results = preprocess_features(self._input_features, new_dataset)
        for feature_name, _data, artifact in input_results:
            self._register_artifact(feature_name, artifact)
        self._input_vectors = [
            data for (feature_name, data, artifact) in input_results
        ]

    def make_predictions(self, new_dataset: Dataset) -> np.ndarray:
        """Make predictions for new data.

        Args:
            new_data (Dataset):

        Returns:
            np.ndarray: The predictions for the new dataset.
        """
        # Make sure the right features are in the correct order
        self._validate_prediction_features(new_dataset)
        self._preprocess_prediction_columns(new_dataset)

        observations = self._compact_vectors(self._input_vectors)
        encoder = self._artifacts[self._target_feature.name]
        predictions = self._model.predict(observations)
        if encoder.__class__.__name__ == "StandardScaler":
            # Return the original values using the encoder
            return encoder.inverse_transform(predictions)
        # Return the original labels using the encoder
        return [encoder.categories_[0][index] for index in predictions]
