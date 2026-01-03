import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import streamlit as st
import subprocess
import sys

#FIX FOR STREAMLIT IMPORT ERRORS
def ensure_dependencies():
    """Forces the environment to recognize sentence-transformers if the loader fails."""
    try:
        import sentence_transformers
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers"])

ensure_dependencies()
load_dotenv()

class Config:
    try:
        XAI_API_KEY = st.secrets["XAI_API_KEY"]
    except (KeyError, FileNotFoundError, getattr(st.errors, 'StreamlitSecretNotFoundError', Exception)):
        XAI_API_KEY = os.getenv("XAI_API_KEY")
    
    MODEL_NAME = "llama-3.1-8b-instant"
    
    # IMPROVEMENT: Reduced chunk size and increased overlap 
    # This helps the model "see" more specific details like tech stacks.
    CHUNK_SIZE = 500 
    CHUNK_OVERLAP = 100

    @staticmethod
    def get_llm():
        return ChatOpenAI(
            openai_api_base="https://api.groq.com/openai/v1",
            openai_api_key=Config.XAI_API_KEY,
            model_name=Config.MODEL_NAME,
            temperature=0
        )
