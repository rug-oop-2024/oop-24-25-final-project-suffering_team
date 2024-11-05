import io
import pickle

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.metric import (
    CLASSIFICATION_METRICS,
    REGRESSION_METRICS,
    get_metric,
)
from autoop.core.ml.model import get_model
from autoop.core.ml.pipeline import Pipeline
from autoop.functional.feature import detect_feature_types

import pandas as pd
import streamlit as st

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
    # Load the pipeline data
    for pipeline in pipelines:
        if pipeline.name == name:
            correct_pipeline = pipeline
            break
    st.write(correct_pipeline.id)
    pipeline_setup = automl.registry.get(correct_pipeline.id)
    pipeline_data = pickle.loads(pipeline_setup.data)

    for key in pipeline_setup.metadata:
        if "pipeline_model" in key:
            model_id = pipeline_setup.metadata[key]
            break
    model_setup = automl.registry.get(model_id)
    model_data = pickle.loads(model_setup.data)
    model = get_model(model_data["model"])
    model._parameters = model_data["parameters"]
    model._n_features = model_data["features"]
    model._fitted = model_data["fitted"]

    pipeline_setup.data = pipeline_data
    pipeline_setup.metadata[key] = model

    input_features = pipeline_data.get("input_features")
    target_feature = pipeline_data.get("target_feature")
    metrics = pipeline_data.get("metrics")
    split = pipeline_data.get("split")
    dataset = pipeline_data.get("dataset")

    st.write("## Pipeline Summary:")
    st.write("The following pipeline has been created:")
    st.write("- **Dataset**:", dataset.name)
    st.write("- **Target Feature**:", target_feature.name)
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

    new_dataset = st.selectbox(
        "Choose dataset to use on model or upload your own in datasets page:",
        (dataset.name for dataset in datasets),
        index=None,
    )

    if new_dataset is not None:
        for dataset in datasets:
            if dataset.name == new_dataset:
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
        model_type = model.type
        possible_features = detect_feature_types(correct_dataset)
        if model_type == "regression":
            correct_features = [
                feature
                for feature in possible_features
                if feature.type == "numerical"
            ]
        else:
            correct_features = [
                feature
                for feature in possible_features
                if feature.type == "categorical"
            ]
        st.write("## Feature Selection:")
        target = st.selectbox(
            "Select target column for prediction:",
            correct_features,
            index=None,
        )
        if target is not None:
            input_features_options = [
                feature
                for feature in possible_features
                if feature.name != target.name
            ]
            new_input_features = st.multiselect(
                "Select input columns for model:", input_features_options
            )
            st.write(
                "Chosen columns:",
                ", ".join(feature.name for feature in input_features),
            )
            # As we only want to predict the split should be 0
            new_split = 0
            loaded_pipeline = Pipeline(
                metrics=metrics,
                dataset=correct_dataset,
                model=model,
                input_features=new_input_features,
                target_feature=target,
                split=new_split,
            )
            st.write(
                "model:",
                model.__class__.__name__,
                "\n column to predict:",
                target.name,
                "\n column used to predict",
                (feature.name for feature in input_features),
                "\n evaluating with:",
                (metric.__class__.__name__ for metric in metrics),
            )
            if st.button("Predict"):
                loaded_pipeline._preprocess_features()
                loaded_pipeline._split_data()
                loaded_pipeline._evaluate()
                st.write(
                    "metrics:",
                    loaded_pipeline._metrics_results,
                    "predictions:",
                    loaded_pipeline._predictions,
                )
