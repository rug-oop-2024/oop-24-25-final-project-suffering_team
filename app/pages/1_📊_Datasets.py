from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

import pandas as pd
import streamlit as st

if "executed_pipeline" in st.session_state:
    st.session_state.result = None
    st.session_state.executed_pipeline = None
    st.session_state.new_predictions = None

automl = AutoMLSystem.get_instance()

st.set_page_config(page_title="Datasets", page_icon="📊")

datasets = automl.registry.list(type="dataset")

st.write("# 📊 Datasets")
st.write(
    "Currently saved datasets:",
    ", ".join(dataset.name for dataset in datasets),
)
csv_file = st.file_uploader("Upload your own csv dataset", ["csv"])

if csv_file is not None:
    dataframe = pd.read_csv(csv_file)
    file_name = csv_file.name

    shuffle_data = st.checkbox("Shuffle the data before saving.", value=False)

    if shuffle_data:
        dataframe = dataframe.sample(frac=1).reset_index(drop=True)
        st.write("The data has been shuffled.")
    else:
        st.write("The data has *not* been shuffled.")

    st.write(dataframe.head())

    dataset = Dataset.from_dataframe(dataframe, file_name, file_name)
    save_button = st.button("Save Dataset")

    if save_button and dataset not in datasets:
        automl._storage.save(dataset.save(dataframe), file_name)
        automl._registry.register(dataset)
        st.success(f"Dataset '{file_name}' has been saved successfully!")
    st.write("Go to the modelling page to use the dataset once saved.")
