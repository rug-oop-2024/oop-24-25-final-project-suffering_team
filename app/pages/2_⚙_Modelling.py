import io
from math import ceil

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

# The lowest number of samples the user can select for training data.
MIN_TRAINING_SAMPLES = 3

st.set_page_config(page_title="Modelling", page_icon="📈")


def write_helper_text(text: str) -> None:
    """Write some text.

    Args:
        text (str): text to be written
    """
    st.write(f'<p style="color: #888;">{text}</p>', unsafe_allow_html=True)


if "executed_pipeline" not in st.session_state:
    st.session_state.result = None
    st.session_state.executed_pipeline = None

if "new_predictions" in st.session_state:
    st.session_state.new_predictions = None

st.write("# ⚙ Modelling")
write_helper_text(
    "".join(("In this section, you can design a ",
    "machine learning pipeline to train a model on a dataset."))
)

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")
pipelines = automl.registry.list(type="pipeline")

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
    # This needs fixing as the storage deletes only the objects.
    st.write("## Delete dataset")
    if st.button("Delete dataset"):
        for dataset in datasets:
            if dataset.name == name:
                automl.registry.delete(dataset.id)
                st.rerun()

    for dataset in datasets:
        if dataset.name == name:
            chosen_data = dataset
            break
    data_bytes = chosen_data.data
    csv = data_bytes.decode()
    csv_data = pd.read_csv(io.StringIO(csv))
    st.write("Chosen data:", csv_data.head())

    correct_dataset = Dataset.from_dataframe(
        name=chosen_data.name,
        data=csv_data,
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
    # There should be at least some training samples
    min_split = ceil(MIN_TRAINING_SAMPLES / len(csv_data) * 100) / 100
    split = st.slider(
        "Select how much of the data is for training.", min_split, 0.99, 0.80
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
        st.session_state.result = pipeline.execute()
        st.session_state.executed_pipeline = pipeline

    if st.session_state.executed_pipeline is not None:
        result = st.session_state.result
        # Extract results
        train_result = result["train_metrics"]
        test_result = result["test_metrics"]
        predictions = result["predictions"]
        # Get the original labels

        st.write("## Last Executed Pipeline Results:")

        st.write("### Train metrics:")
        for metric_result in train_result:
            metric_name = metric_result[0].__class__.__name__
            st.write(f"- **{metric_name}**: {metric_result[1]:.4f}")

        st.write("### Test metrics:")
        for metric_result in test_result:
            metric_name = metric_result[0].__class__.__name__
            st.write(f"- **{metric_name}**: {metric_result[1]:.4f}")

        st.write("### Predictions:")
        result_dataframe = pd.DataFrame(predictions, columns=[target.name])
        if feature_type == "numerical":
            # Limit the number of decimals in the predictions.
            result_dataframe[target.name] = result_dataframe[target.name].map(
                "{:,.4f}".format
            )

        if max_display == 0 or max_display >= len(predictions):
            # Show all predictions
            st.write(result_dataframe)
        else:
            # Show a selection of the predictions
            st.write(result_dataframe.head(max_display))
            st.write(
                f"... and {len(predictions) - max_display} ",
                "more.",
            )

        # Allow the user to download all predictions
        csv_data = result_dataframe.to_csv()
        st.download_button(
            label="Download Predictions as csv.",
            data=csv_data,
            file_name="predictions.csv",
            mime="text/csv",
        )

        # The pipeline needs to have a trained model before it can be saved.
        st.write("## Save Last Executed Pipeline:")
        pipeline_name = st.text_input("Give name to pipeline:", "MyPipeline")
        pipeline_version = st.text_input(
            "Give the version of the pipeline", "1.0.0"
        )
        if (pipeline_name, pipeline_version) in (
            (pipeline.name, pipeline.version) for pipeline in pipelines
        ):
            st.write(
                "This name and version is already saved\n",
                "Saved pipelines:",
                ((pipeline.name, pipeline.version) for pipeline in pipelines),
            )
        else:
            if st.button("Save pipeline"):
                pipeline = st.session_state.executed_pipeline
                all_artifacts = pipeline.artifacts
                for artifact in all_artifacts:
                    if artifact.name == "pipeline_config":
                        artifact.name = pipeline_name
                        artifact.asset_path = (
                            f"{pipeline_name}-{pipeline_version}"
                        )
                        artifact.version = pipeline_version
                        artifact.type = "pipeline"

                        encoded_path = artifact._base64_encode(
                            artifact.asset_path
                        )
                        artifact.id = f"{encoded_path}-{artifact.version}"

                        pipeline_artifact = artifact
                    elif artifact.type == "model":
                        artifact.version = pipeline_version
                        artifact.asset_path = "-".join(
                            (
                                f"{pipeline_name}",
                                f"{pipeline_version}",
                                f"{artifact.name}",
                            )
                        )
                        encoded_path = artifact._base64_encode(
                            artifact.asset_path
                        )
                        artifact.id = f"{encoded_path}-{artifact.version}"
                        automl._registry.register(artifact)
                    else:
                        automl._registry.register(artifact)
                for artifact in all_artifacts:
                    if artifact.type != "pipeline":
                        pipeline_artifact.save_metadata(artifact)
                automl._registry.register(pipeline_artifact)
                st.success(
                    f"Pipeline '{pipeline_name}' has been saved successfully!"
                )
