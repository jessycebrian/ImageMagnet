import streamlit as st
from demo_streamlit import get_images_from_query


# Create a Streamlit widget for inputting vectors
st.title("ImageMagnet Demo")

query = st.text_input("What are you looking for in your dream home?")
num_results = st.slider("Number of results to display", min_value=1, max_value=472, value=5)

if query:   
    st.image(get_images_from_query(query,top_k = num_results))