import streamlit as st

if "executed_pipeline" in st.session_state:
    st.session_state.result = None
    st.session_state.executed_pipeline = None
    st.session_state.new_predictions = None

st.set_page_config(
    page_title="Instructions",
    page_icon="👋",
)

st.markdown(open("INSTRUCTIONS.md").read())
