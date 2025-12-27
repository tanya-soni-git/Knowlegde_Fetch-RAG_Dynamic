from langgraph.graph import StateGraph, END
from src.state.rag_state import RAGState
from src.node.reactnode import RAGNodes

class GraphBuilder:
    def __init__(self, retriever, llm):
        self.nodes = RAGNodes(retriever, llm)
        self.graph = None

    def build(self):
        workflow = StateGraph(RAGState)
        workflow.add_node("retrieve", self.nodes.retrieve_docs)
        workflow.add_node("respond", self.nodes.generate_answer)
        
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "respond")
        workflow.add_edge("respond", END)
        
        self.graph = workflow.compile()

    def run(self, question: str):
        return self.graph.invoke({"question": question, "retrieved_docs": []})
