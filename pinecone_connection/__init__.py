import logging 


from pinecone import Pinecone
import streamlit as st

API_KEY = st.secrets["API_KEY"]
HOST = st.secrets["HOST"]
INDEX_NAME = st.secrets["INDEX_NAME"]

pc = Pinecone(
        api_key = API_KEY
    )
right_index = pc.Index(index_name = INDEX_NAME, host = HOST)

logging.info(f'Conected to Pinecone index: {right_index.describe_index_stats}')
