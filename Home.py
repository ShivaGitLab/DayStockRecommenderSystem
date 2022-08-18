import streamlit as st
from PIL import Image

st.set_page_config(
    page_title='Day Stock Recommender App',
    page_icon="❄",
    layout='wide'
)

st.sidebar.success("Select a page above")
st.markdown("<h1 style='text-align: center; color: SkyBlue;'> Day Stock Recommendor System - Demo</h1>",
            unsafe_allow_html=True)

st.sidebar.image("Pictures\Logo.png", use_column_width=True)

with open("style.css") as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

image = Image.open('Pictures\Market.jpg')
st.image(image, width = 1000)


