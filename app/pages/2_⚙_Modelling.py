import io

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.metric import (
    CLASSIFICATION_METRICS,
    REGRESSION_METRICS,
    get_metric,
)
from autoop.core.ml.model import (
    CLASSIFICATION_MODELS,
    REGRESSION_MODELS,
    get_model,
)
from autoop.core.ml.pipeline import Pipeline
from autoop.functional.feature import detect_feature_types

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Modelling", page_icon="📈")


def write_helper_text(text: str):
    """Write some text.

    Args:
        text (str): text to be written
    """
    st.write(f'<p style="color: #888;">{text}</p>', unsafe_allow_html=True)


st.write("# ⚙ Modelling")
write_helper_text(
    "In this section, you can design a "
    + "machine learning pipeline to train a model on a dataset."
)

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")

metrics = []
selected_metrics = False
selected_features = False
selected_model = False

st.write("## Dataset Selection:")
name = st.selectbox(
    "Choose dataset to use on model or upload your own in datasets page:",
    (dataset.name for dataset in datasets),
    index=None,
)

if name is not None:
    for dataset in datasets:
        if dataset.name == name:
            chosen_data = dataset
            break
    data_bytes = chosen_data.data
    csv = data_bytes.decode()
    full_data = pd.read_csv(io.StringIO(csv))
    st.write("Chosen data:", full_data.head())
    correct_dataset = Dataset.from_dataframe(
        name=chosen_data.name,
        data=full_data,
        asset_path=chosen_data.asset_path,
        version=chosen_data.version,
    )
    features = detect_feature_types(correct_dataset)
    st.write("## Feature Selection:")
    target = st.selectbox(
        "Select target column for prediction:", features, index=None
    )
    if target is not None:
        input_features_options = [
            feature for feature in features if feature.name != target.name
        ]
        st.write(f"Target column: {target.name}")
        input_features = st.multiselect(
            "Select input columns for model:", input_features_options
        )
        st.write(
            "Chosen columns:",
            ", ".join(feature.name for feature in input_features),
        )
        if len(input_features) >= 1:
            selected_features = True
        feature_type = target.type

        st.write("## Model Selection:")
        if feature_type == "numerical":
            st.write("Task type is regression.")
            model = st.selectbox("Choose models to use:", REGRESSION_MODELS)
            model = get_model(model)
            if model is not None:
                selected_model = True

            metrics = []
            metric_names = st.multiselect(
                "Select metrics to evaluate:", REGRESSION_METRICS
            )
            metrics = [get_metric(metric) for metric in metric_names]
            selected_metrics = len(metrics) >= 1
        elif feature_type == "categorical":
            st.write("Task type is classification.")
            model = st.selectbox(
                "Choose models to use:", CLASSIFICATION_MODELS
            )
            model = get_model(model)
            if model is not None:
                selected_model = True

            metrics = []
            metric_names = st.multiselect(
                "Select metrics to evaluate:", CLASSIFICATION_METRICS
            )
            metrics = [get_metric(metric) for metric in metric_names]
            selected_metrics = len(metrics) >= 1


if selected_model and selected_metrics and selected_features:
    split = st.slider(
        "Select how much of the data is for training.", 0.01, 0.99, 0.80
    )
    pipeline = Pipeline(
        metrics=metrics,
        dataset=correct_dataset,
        model=model,
        input_features=input_features,
        target_feature=target,
        split=split,
    )
    # Pipeline summary
    st.write("## Pipeline Summary:")
    st.write("The following pipeline has been created:")
    st.write("- **Dataset**:", correct_dataset.name)
    st.write("- **Target Feature**:", target.name)
    st.write(
        "- **Input Features**:",
        ", ".join(feature.name for feature in input_features),
    )
    st.write("- **Model**:", model.__class__.__name__)
    st.write(
        "- **Metrics**:",
        ", ".join(metric.__class__.__name__ for metric in metrics),
    )
    st.write("- **Training Split**:", f"{split:.0%} of data")

    # Not all predictions should be showed if there are many predictions
    max_display = st.number_input(
        "Enter the maximum number of predictions to display. (0=all))",
        min_value=0,
        value=50,
        step=1,
    )
    if st.button("Execute pipeline"):
        result = pipeline.execute()

        # Extract results
        train_result = result["train_metrics"]
        test_result = result["test_metrics"]
        predictions = result["predictions"]
        # Get the original labels
        if target.type == "categorical":
            unique_target_values = full_data[target.name].unique()
            predictions = [unique_target_values[pred] for pred in predictions]

        st.write("## Pipeline Results:")

        st.write("### Train metrics:")
        for metric_result in train_result:
            metric_name = metric_result[0].__class__.__name__
            st.write(f"- **{metric_name}**: {metric_result[1]:.4f}")

        st.write("### Test metrics:")
        for metric_result in test_result:
            metric_name = metric_result[0].__class__.__name__
            st.write(f"- **{metric_name}**: {metric_result[1]:.4f}")

        st.write("### Predictions:")
        if max_display == 0 or max_display >= len(predictions):
            # Show all predictions
            st.code(predictions)
        else:
            # Show a selection of the predictions
            show_predictions = predictions[:max_display]
            st.code(show_predictions)
            st.write(
                f"... and {len(predictions) - max_display} ",
                "more.",
            )

    # The pipeline needs to have a trained model before it can be saved.
    st.write("## Save Pipeline:")
    pipeline_name = st.text_input("Give name to pipeline:", "MyPipeline")
    pipeline_version = st.text_input(
        "Give the version of the pipeline", "1.0.0"
    )
    if st.button("Save pipeline"):
        all_artifacts = pipeline.artifacts
        for artifact in all_artifacts:
            if artifact.name == "pipeline_config":
                artifact.name = pipeline_name
                artifact.asset_path = pipeline_name
                artifact.version = pipeline_version
                artifact.type = "pipeline"

                encoded_path = artifact._base64_encode(artifact.asset_path)
                artifact.id = f"{encoded_path}-{artifact.version}"

                pipeline_artifact = artifact
            else:
                automl._registry.register(artifact)
                pipeline_artifact.save_metadata(artifact)
        automl._registry.register(pipeline_artifact)
        st.write(pipeline_artifact)
