import os
import tempfile
import streamlit as st
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def process_uploaded_file(self, uploaded_file):
        """Saves memory-based file to a temp path for loading."""
        # Create a temp file to store the upload so LangChain loaders can read it
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            # Choose loader based on file extension
            if uploaded_file.name.lower().endswith('.pdf'):
                loader = PyPDFLoader(tmp_path)
            else:
                loader = TextLoader(tmp_path)
            
            # 1. Loading logic
            docs = loader.load()
            
            # Handle empty extractions (e.g., scanned PDFs without text)
            if not docs:
                return [] 
            
            # 2. Splitting logic
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size, 
                chunk_overlap=self.chunk_overlap
            )
            return text_splitter.split_documents(docs)
            
        except Exception as e:
            # Error handling for the UI
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            return []
            
        finally:
            # Clean up the temp file after processing
            if os.path.exists(tmp_path):
                os.remove(tmp_path)