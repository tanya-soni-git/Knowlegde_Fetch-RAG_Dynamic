import os
import tempfile
import streamlit as st
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    WebBaseLoader,
    WikipediaLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class DocumentProcessor:
    """
    Handles ingestion and chunking of multiple data sources
    for building a RAG knowledge base.
    """

    def __init__(self, chunk_size=300, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # -------------------- COMMON SPLITTER --------------------
    def split_documents(self, docs):
        if not docs:
            return []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        return splitter.split_documents(docs)

    # -------------------- FILE UPLOAD --------------------
    def process_uploaded_file(self, uploaded_file):
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=Path(uploaded_file.name).suffix
        ) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            if uploaded_file.name.lower().endswith(".pdf"):
                loader = PyPDFLoader(tmp_path)
            elif uploaded_file.name.lower().endswith(".csv"):
                loader = CSVLoader(tmp_path)
            else:
                loader = TextLoader(tmp_path, encoding="utf-8")

            docs = loader.load()
            return self.split_documents(docs)

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            return []

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # -------------------- WEB URL --------------------
    def process_url(self, url):
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            return self.split_documents(docs)

        except Exception as e:
            st.error(f"Error loading URL: {str(e)}")
            return []

    # -------------------- PASTE TEXT --------------------
    def process_raw_text(self, text, source="manual_text"):
        """
        Handles copy-pasted text (articles, notes, transcripts, etc.)
        """
        if not text or not text.strip():
            return []

        docs = [
            Document(
                page_content=text.strip(),
                metadata={
                    "source": source,
                    "type": "manual"
                }
            )
        ]

        return self.split_documents(docs)

    # -------------------- WIKIPEDIA --------------------
    def process_wikipedia(self, query):
        """
        Accepts:
        - Topic name (e.g. 'Java Virtual Machine')
        - Full Wikipedia URL
        """
        try:
            if "wikipedia.org/wiki/" in query:
                query = query.split("/wiki/")[-1].replace("_", " ")

            loader = WikipediaLoader(
                query=query,
                load_max_docs=1
            )

            docs = loader.load()
            return self.split_documents(docs)

        except Exception as e:
            st.error(f"Error loading Wikipedia content: {str(e)}")
            return []
