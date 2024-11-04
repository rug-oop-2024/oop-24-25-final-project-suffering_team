import pickle

from app.core.system import AutoMLSystem
from autoop.core.ml.metric import CLASSIFICATION_METRICS, REGRESSION_METRICS
from autoop.core.ml.model import get_model
from autoop.core.ml.pipeline import Pipeline

import pandas as pd
import streamlit as st

automl = AutoMLSystem.get_instance()

st.set_page_config(page_title="Deployment", page_icon="🛠️")

pipelines = automl.registry.list(type="pipeline")

st.write("# Pipeline Selection:")
name = st.selectbox(
    "Choose pipeline to use to predict on new dataset:",
    (pipeline.name for pipeline in pipelines),
    index=None,
)

if name is not None:
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
    st.write("Selected pipeline:", pipeline_setup.data)
