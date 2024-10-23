import pickle
from typing import List

import numpy as np

from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric
from autoop.core.ml.model import Model
from autoop.functional.preprocessing import preprocess_features


class Pipeline:
    """A pipeline class to bring together different dataprocessing classes."""

    def __init__(
        self,
        metrics: List[Metric],
        dataset: Dataset,
        model: Model,
        input_features: List[Feature],
        target_feature: Feature,
        split=0.8,
    ):
        """Initialize the pipeline.

        Args:
            metrics (List[Metric]): The list of metrics to be used.
            dataset (Dataset): The data used for processing.
            model (Model): The model used to process the data.
            input_features (List[Feature]): A list of features stating
                their name and whether they are numerical or categorical.
            target_feature (Feature): The feature to make predictions for.
            split (float, optional): The distribution of the train and testing data.
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
        if (
            target_feature.type == "categorical"
            and model.type != "classification"
        ):
            raise ValueError(
                "Model type must be classification for categorical target feature"
            )
        if target_feature.type == "continuous" and model.type != "regression":
            raise ValueError(
                "Model type must be regression for continuous target feature"
            )

    def __str__(self) -> str:
        """Return a string representation of the most important attributes."""
        return f"""
Pipeline(
    model={self._model.type},
    input_features={list(map(str, self._input_features))},
    target_feature={str(self._target_feature)},
    split={self._split},
    metrics={list(map(str, self._metrics))},
)
"""

    @property
    def model(self) -> Model:
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
            "input_features": self._input_features,
            "target_feature": self._target_feature,
            "split": self._split,
        }
        artifacts.append(
            Artifact(name="pipeline_config", data=pickle.dumps(pipeline_data))
        )
        artifacts.append(
            self._model.to_artifact(name=f"pipeline_model_{self._model.type}")
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
        input_results = preprocess_features(self._input_features, self._dataset)
        for feature_name, data, artifact in input_results:
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
            vector[: int(split * len(vector))] for vector in self._input_vectors
        ]
        self._test_X = [
            vector[int(split * len(vector)) :] for vector in self._input_vectors
        ]
        self._train_y = self._output_vector[
            : int(split * len(self._output_vector))
        ]
        self._test_y = self._output_vector[
            int(split * len(self._output_vector)) :
        ]

    def _compact_vectors(self, vectors: List[np.array]) -> np.array:
        """Combine a list of vectors into one array."""
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
        self._predictions = predictions

    def execute(self) -> dict[str, List]:
        """Process the data in the model and collect the results."""
        self._preprocess_features()
        self._split_data()
        self._train()
        self._evaluate()
        return {
            "train_metrics": self._train_metrics_results,
            "test_metrics": self._metrics_results,
            "predictions": self._predictions,
        }
