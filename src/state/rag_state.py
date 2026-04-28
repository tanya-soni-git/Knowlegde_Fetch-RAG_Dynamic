from typing import List, TypedDict
from langchain_core.documents import Document

class RAGState(TypedDict):
    question: str
    retrieved_docs: List[Document]
    answer: str
    suggested_questions: List[str]  # New field added for auto-suggestions
