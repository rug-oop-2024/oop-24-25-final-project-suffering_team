from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

import pandas as pd
import streamlit as st

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")

st.write(
    "Currently saved datasets:",
    ", ".join(dataset.name for dataset in datasets),
)
csv_file = st.file_uploader("Upload your own csv dataset", ["csv"])

if csv_file is not None:
    dataframe = pd.read_csv(csv_file)
    file_name = csv_file.name

    # Shuffle the data
    dataframe = dataframe.sample(n=len(dataframe))
    # Reassigns the indices
    dataframe = dataframe.reset_index(drop=True)

    st.write(dataframe.head())
    dataset = Dataset.from_dataframe(dataframe, file_name, file_name)
    if dataset not in datasets:
        automl._storage.save(dataset.save(dataframe), file_name)
        automl._registry.register(dataset)
    st.write("Go to modelling page to use the dataset.")
