import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import streamlit as st

load_dotenv()

class Config:
    # Prioritizes Streamlit Cloud secrets, falls back to local .env
    # This version won't crash if st.secrets is empty
    try:
        XAI_API_KEY = st.secrets["XAI_API_KEY"]
    except (KeyError, FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        XAI_API_KEY = os.getenv("XAI_API_KEY")
    
    MODEL_NAME = "llama-3.1-8b-instant"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    @staticmethod
    def get_llm():
        return ChatOpenAI(
            openai_api_base="https://api.groq.com/openai/v1",
            openai_api_key=Config.XAI_API_KEY,
            model_name=Config.MODEL_NAME,
            temperature=0
        )