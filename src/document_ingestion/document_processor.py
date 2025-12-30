import os
import tempfile
import streamlit as st
from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    CSVLoader, 
    WebBaseLoader, 
    YoutubeLoader, 
    WikipediaLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentProcessor:
    def __init__(self, chunk_size=500, chunk_overlap=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, docs):
        """Helper to split loaded documents into chunks."""
        if not docs:
            return []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        return text_splitter.split_documents(docs)

    def process_uploaded_file(self, uploaded_file):
        """Handles PDF, TXT, and CSV uploads."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            if uploaded_file.name.lower().endswith('.pdf'):
                loader = PyPDFLoader(tmp_path)
            elif uploaded_file.name.lower().endswith('.csv'):
                loader = CSVLoader(tmp_path)
            else:
                loader = TextLoader(tmp_path)
            
            docs = loader.load()
            return self.split_documents(docs)
            
        except Exception as e:
            st.error(f"Error processing file {uploaded_file.name}: {str(e)}")
            return []
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def process_url(self, url):
        """Handles direct Website URLs."""
        try:
            loader = WebBaseLoader(url)
            return self.split_documents(loader.load())
        except Exception as e:
            st.error(f"Error loading URL: {str(e)}")
            return []

    def process_youtube(self, url):
        """Handles YouTube Video Transcripts."""
        try:
            loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
            return self.split_documents(loader.load())
        except Exception as e:
            st.error(f"Error loading YouTube transcript: {str(e)}")
            return []

    def process_wikipedia(self, query):
        """Handles Wikipedia Topic Searches."""
        try:
            loader = WikipediaLoader(query=query, load_max_docs=1)
            return self.split_documents(loader.load())
        except Exception as e:
            st.error(f"Error loading Wikipedia: {str(e)}")
            return []
