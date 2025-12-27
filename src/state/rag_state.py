from typing import List, TypedDict
from langchain_core.documents import Document

class RAGState(TypedDict):
    question: str
    retrieved_docs: List[Document]
    answer: str