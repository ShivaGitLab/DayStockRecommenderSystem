import streamlit as st
from PIL import Image

st.set_page_config(
    page_title='Day Stock Predictor App',
    page_icon="❄",
    layout='wide'
)

st.sidebar.success("Select a page above")
st.markdown("<h1 style='text-align: center; color: Gold;'>Day Stock Recommendor System - Demo</h1>",
            unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.write(' ')
with col2:
    image = Image.open('Pictures\Logo.png')
    st.image("Pictures\Logo.png", caption='University of Maryland - Baltimore County')
with col3:
    st.write(' ')


