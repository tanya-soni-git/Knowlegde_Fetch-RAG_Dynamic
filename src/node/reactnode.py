from langchain_core.messages import HumanMessage, SystemMessage
from src.state.rag_state import RAGState


class RAGNodes:
    """
    Simple, stable RAG node implementation.
    - No ReAct
    - No tools
    - No agent state leakage
    - Safe for Streamlit + LangGraph
    """

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

    # 1️⃣ Retrieve documents
    def retrieve_docs(self, state: RAGState):
        """
        Fetch relevant documents for the user question.
        """
        question = state["question"]
        docs = self.retriever.invoke(question)

        return {
            "retrieved_docs": docs
        }

    # 2️⃣ Generate answer
    def generate_answer(self, state: RAGState):
        """
        Generate answer strictly from retrieved documents.
        """
        question = state["question"]
        docs = state.get("retrieved_docs", [])

        if not docs:
            return {
                "answer": "I could not find relevant information in the uploaded documents."
            }

        # Combine document content
        context = "\n\n".join(
            f"Document {i+1}:\n{doc.page_content}"
            for i, doc in enumerate(docs)
        )

        system_message = SystemMessage(
            content=(
                "You are a helpful assistant.\n"
                "Answer the question using ONLY the provided document context.\n"
                "If the answer is not present in the documents, say so clearly.\n"
                "Be concise and accurate."
            )
        )

        human_message = HumanMessage(
            content=(
                f"Context:\n{context}\n\n"
                f"Question:\n{question}"
            )
        )

        response = self.llm.invoke(
            [system_message, human_message]
        )

        return {
            "answer": response.content
        }

    # -------------------------
    # 3️⃣ Generate Auto-Suggestions
    # -------------------------
    def generate_suggestions(self, state: RAGState):
        """
        Generates 5 relevant questions based on context or the last answer.
        """
        docs = state.get("retrieved_docs", [])
        if not docs:
            return {"suggested_questions": []}

        # Combine retrieved document content for suggestion context
        context = "\n\n".join([doc.page_content for doc in docs])
        
        prompt = f"""Based on the following context, generate 5 concise, 
        engaging questions that a user might want to ask. 
        Return ONLY a bulleted list of questions.
        
        Context: {context}"""
        
        response = self.llm.invoke(prompt)
        
        # Parse the bulleted list into a Python list
        # Handles stripping common bullet formats like "-", "1.", etc.
        questions = [
            q.strip("- ").strip("12345. ").strip() 
            for q in response.content.strip().split("\n") 
            if q.strip()
        ][:5]
        
        return {
            "suggested_questions": questions
        }
