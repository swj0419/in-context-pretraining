import streamlit as st

st.markdown(
    """
            
# Home Page for Retro Dataset Viewer

This streamlit app makes it easy to inspect the input data to Retro.
There are two sub-apps, browsable on the left tab:

1. Raw: This allows free exploration of the raw data with minimal processing.
2. Retro Input: This allows exploration of the data that is fed into Retro at train/inference time

The RETRO model has a few components, shown below (and explained below the image)
"""
)
st.image("RETRO.png", caption="RETRO Model Illustration")
