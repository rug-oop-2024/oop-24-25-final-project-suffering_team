import io
import pickle

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.model.model import Model
from autoop.core.ml.pipeline import Pipeline

import pandas as pd
import streamlit as st

from exceptions import DatasetValidationError

if "executed_pipeline" in st.session_state:
    st.session_state.result = None
    st.session_state.executed_pipeline = None

automl = AutoMLSystem.get_instance()

st.set_page_config(page_title="Deployment", page_icon="🚀")

pipelines = automl.registry.list(type="pipeline")
datasets = automl.registry.list(type="dataset")

st.write("# Pipeline Selection:")
name = st.selectbox(
    "Choose pipeline to use to predict on new dataset:",
    (pipeline.name for pipeline in pipelines),
    index=None,
)

if name is not None:
    # This needs fixing as the storage deletes only the objects.
    st.write("## Delete pipeline")
    if st.button("Delete pipeline"):
        for pipeline in pipelines:
            if pipeline.name == name:
                pipeline_to_delete = pipeline
                break
        for artifact_id in pipeline_to_delete.metadata.values():
            automl.registry.delete(artifact_id)
        automl.registry.delete(pipeline.id)
        st.rerun()

    # Load the pipeline data
    for pipeline in pipelines:
        if pipeline.name == name:
            correct_pipeline = pipeline
            break
    st.write(correct_pipeline.id)
    pipeline_setup = automl.registry.get(correct_pipeline.id)
    pipeline_data = pickle.loads(pipeline_setup.data)
    target_feature = pipeline_data.get("target_feature")

    for key in pipeline_setup.metadata:
        if "pipeline_model" in key:
            model_id = pipeline_setup.metadata[key]
        elif key == target_feature.name:
            target_id = pipeline_setup.metadata[key]
    model_setup = automl.registry.get(model_id)
    recreated_model = Model.from_artifact(model_setup)

    target_setup = automl.registry.get(target_id)
    encoder = pickle.loads(target_setup.data)
    pipeline_setup.data = pipeline_data
    pipeline_setup.metadata[key] = recreated_model

    input_features = pipeline_data.get("input_features")
    metrics = pipeline_data.get("metrics")
    split = pipeline_data.get("split")
    old_dataset = pipeline_data.get("dataset")

    st.write("## Pipeline Summary:")
    st.write("The following pipeline has been created:")
    st.write("- **Dataset**:", old_dataset.name)
    st.write("- **Target Feature**:", target_feature.name)
    st.write(
        "- **Input Features**:",
        ", ".join(feature.name for feature in input_features),
    )
    st.write("- **Model**:", recreated_model.__class__.__name__)
    st.write(
        "- **Metrics**:",
        ", ".join(metric.__class__.__name__ for metric in metrics),
    )
    st.write("- **Training Split**:", f"{split:.0%} of data")

    if "train_results" in pipeline_data:
        train_results = pipeline_data.get("train_results")
        test_results = pipeline_data.get("test_results")
        st.write("### Train metrics:")
        for metric_result in train_results:
            metric_name = metric_result[0].__class__.__name__
            st.write(f"- **{metric_name}**: {metric_result[1]:.4f}")

        st.write("### Test metrics:")
        for metric_result in test_results:
            metric_name = metric_result[0].__class__.__name__
            st.write(f"- **{metric_name}**: {metric_result[1]:.4f}")

    # Tell the user how to format the input columns
    st.write("## Upload New Data")
    st.write("The dataset should contain the following input columns:")
    st.write(
        f"- '{input_feature.name}' which is {input_feature.type},  \n"
        for input_feature in input_features
    )

    # Load the data set the user chooses
    new_dataset = st.selectbox(
        "Choose dataset to make predictions for:",
        (dataset.name for dataset in datasets),
        index=None,
    )
    if new_dataset is not None:
        for dataset in datasets:
            if dataset.name == new_dataset:
                chosen_data = dataset
                break
        # Decode the new data set
        data_bytes = chosen_data.data
        csv = data_bytes.decode()
        full_data = pd.read_csv(io.StringIO(csv))

        st.write("Chosen data:", full_data.head())
        new_dataset = Dataset.from_dataframe(
            name=chosen_data.name,
            data=full_data,
            asset_path=chosen_data.asset_path,
            version=chosen_data.version,
        )
        loaded_pipeline = Pipeline(
            metrics=[],
            dataset=old_dataset,
            model=recreated_model,
            input_features=input_features,
            target_feature=target_feature,
            split=0,
        )
        loaded_pipeline._register_artifact(target_feature.name, encoder)
        if st.button("Predict"):
            try:
                predictions = loaded_pipeline.make_predictions(new_dataset)
                if recreated_model.type == "classification":
                    # Decode the old data set to find the unique labels
                    data_bytes = old_dataset.data
                    csv = data_bytes.decode()
                    old_data = pd.read_csv(io.StringIO(csv))
                    unique_target_values = old_data[
                        target_feature.name
                    ].unique()
                    predictions = [
                        unique_target_values[pred] for pred in predictions
                    ]
                st.write(predictions)
            except DatasetValidationError as e:
                st.write(str(e))
