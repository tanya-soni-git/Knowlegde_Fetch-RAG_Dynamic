from langgraph.graph import StateGraph, END
from src.state.rag_state import RAGState
from src.node.reactnode import RAGNodes

class GraphBuilder:
    def __init__(self, retriever, llm):
        """
        Initializes the GraphBuilder with necessary nodes and state.
        """
        self.nodes = RAGNodes(retriever, llm)
        self.graph = None

    def build(self):
        """
        Constructs the LangGraph workflow with retrieval, response, and suggestion nodes.
        """
        workflow = StateGraph(RAGState)
        
        # Define the nodes
        workflow.add_node("retrieve", self.nodes.retrieve_docs)
        workflow.add_node("respond", self.nodes.generate_answer)
        workflow.add_node("suggest", self.nodes.generate_suggestions) # New suggestion node
        
        # Set the entry point and define the sequential edges
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "respond")
        workflow.add_edge("respond", "suggest") # Route through suggestions after generating the answer
        workflow.add_edge("suggest", END)
        
        self.graph = workflow.compile()

    def run(self, question: str):
        """
        Invokes the compiled graph with the user's question.
        """
        return self.graph.invoke({"question": question, "retrieved_docs": []})
