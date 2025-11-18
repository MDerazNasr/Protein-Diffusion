"""Demo application."""

import streamlit as st

st.title("Protein Diffusion Demo")
st.write("This interface will generate protein backbones using a diffusion model")
#write adds a simple text block under the title
#st.write is flexible, can display markdown, html, images, DataFrames...
if st.button("Generate Protein"): #draws button
    st.info("Generation pipeline not implemented yet. Check back when model is ready.")
#info is a Special styled message box. --> in blue highlighted box