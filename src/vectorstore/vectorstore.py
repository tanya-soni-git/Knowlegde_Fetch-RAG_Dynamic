from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import subprocess
import sys

class VectorStore:
    def __init__(self):
        try:
            import sentence_transformers
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers"])
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = None

    def create_vectorstore(self, documents):
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        return self.vectorstore

    def get_retriever(self):
        if not self.vectorstore:
            return None
        return self.vectorstore.as_retriever(search_kwargs={"k": 5})
