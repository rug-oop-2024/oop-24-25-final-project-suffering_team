import pandas as pd
import streamlit as st

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")

csv_file = st.file_uploader("Upload your own csv dataset", ["csv"])

if csv_file is not None:
    dataframe = pd.read_csv(csv_file)
    file_name = csv_file.name
    st.write(dataframe.head())
    dataset = Dataset.from_dataframe(dataframe, file_name, file_name)
    automl._storage.save(dataset.save(dataframe), file_name)
    automl._registry.register(dataset)
