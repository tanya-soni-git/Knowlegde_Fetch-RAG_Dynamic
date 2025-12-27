from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import subprocess
import sys

class VectorStore:
    def __init__(self):
        # --- CRITICAL FIX FOR STREAMLIT IMPORT ERROR ---
        # This forces the environment to acknowledge the package at runtime
        try:
            import sentence_transformers
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers"])
        # -----------------------------------------------

        # Local embedding model
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = None

    def create_vectorstore(self, documents):
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        return self.vectorstore

    def get_retriever(self):
        if not self.vectorstore:
            return None
        # --- IMPROVEMENT FOR ACCURACY ---
        # Increasing 'k' from 3 to 5 allows the model to see more chunks of your 
        # document, significantly increasing the chance of finding the "tech stack".
        return self.vectorstore.as_retriever(search_kwargs={"k": 5})