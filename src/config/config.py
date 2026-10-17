import os
import subprocess
import sys
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import streamlit as st

# FIX FOR STREAMLIT IMPORT ERRORS
def ensure_dependencies():
    """Forces the environment to recognize sentence-transformers if the loader fails."""
    try:
        import sentence_transformers
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers"])

ensure_dependencies()
load_dotenv()

def _get_api_key():
    key = os.getenv("GROQ_API_KEY") or os.getenv("XAI_API_KEY")
    if key:
        return key
    try:
        return st.secrets.get("GROQ_API_KEY") or st.secrets.get("XAI_API_KEY")
    except Exception:
        return None

class Config:
    GROQ_API_KEY = _get_api_key()

    MODEL_NAME = "openai/gpt-oss-20b"
    CHUNK_SIZE = 500 
    CHUNK_OVERLAP = 100

    @staticmethod
    def get_llm():
        if not Config.GROQ_API_KEY:
            raise ValueError("Groq API key not found. Please set GROQ_API_KEY in your .env file or Streamlit Secrets.")

        return ChatGroq(
            groq_api_key=Config.GROQ_API_KEY,
            model_name=Config.MODEL_NAME,
            temperature=0
        )