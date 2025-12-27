import streamlit as st
from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder

st.set_page_config(page_title="Dynamic Agentic RAG", page_icon="📂", layout="wide")

def main():
    st.title("📂 Dynamic Document Search")
    
    # Initialize session state so the system doesn't reset on every click
    if 'rag' not in st.session_state:
        st.session_state.rag = None

    # Sidebar UI for uploading files
    with st.sidebar:
        st.header("Upload Center")
        uploaded_files = st.file_uploader(
            "Upload PDF or TXT files", 
            type=['pdf', 'txt'], 
            accept_multiple_files=True
        )
        
        if st.button("Build Knowledge Base") and uploaded_files:
            with st.spinner("Indexing documents..."):
                processor = DocumentProcessor()
                all_chunks = []
                
                # 1. Process all files
                for f in uploaded_files:
                    # Safely extend list only if chunks are returned
                    chunks = processor.process_uploaded_file(f)
                    if chunks:
                        all_chunks.extend(chunks)
                
                # 2. Safety check: Only proceed if chunks exist to avoid IndexError
                if not all_chunks:
                    st.error("No text could be extracted from the uploaded files. Please check if they are empty or scanned images.")
                else:
                    # 3. Create the vector store index
                    vs = VectorStore()
                    vs.create_vectorstore(all_chunks)
                    
                    # 4. Build the workflow graph
                    gb = GraphBuilder(vs.get_retriever(), Config.get_llm())
                    gb.build()
                    
                    # 5. Save to session state
                    st.session_state.rag = gb
                    st.success(f"Success! {len(all_chunks)} chunks indexed.")

    # Main Chat Interface
    if st.session_state.rag:
        user_input = st.text_input("Ask a question about your files:")
        if user_input:
            with st.spinner("Analyzing..."):
                response = st.session_state.rag.run(user_input)
                st.markdown("### 🤖 Answer")
                st.write(response["answer"])
                
                with st.expander("🔍 View Retrieved Sources"):
                    for i, doc in enumerate(response.get("retrieved_docs", [])):
                        st.markdown(f"**Source {i+1}:**")
                        st.info(doc.page_content)
    else:
        st.info("👈 Please upload your files in the sidebar to begin.")

if __name__ == "__main__":
    main()