import pickle

from app.core.system import AutoMLSystem
from autoop.core.ml.metric import (
    CLASSIFICATION_METRICS,
    REGRESSION_METRICS,
    get_metric,
)
from autoop.core.ml.model import get_model
from autoop.core.ml.pipeline import Pipeline

import pandas as pd
import streamlit as st

automl = AutoMLSystem.get_instance()

st.set_page_config(page_title="Deployment", page_icon="🚀")

pipelines = automl.registry.list(type="pipeline")

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

    loaded_pipeline = Pipeline(
        metrics=metrics,
        dataset=dataset,
        model=model,
        input_features=input_features,
        target_feature=target_feature,
        split=split,
    )
