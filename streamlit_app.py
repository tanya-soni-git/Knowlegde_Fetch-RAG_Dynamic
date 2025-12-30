import streamlit as st
from langchain_core.documents import Document

# Corrected Imports
from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Dynamic RAG Pro",
    page_icon="📂",
    layout="wide"
)

def apply_custom_design():
    """Injects CSS for the dark, futuristic 'Pitch Deck' look."""
    st.markdown("""
        <style>
        /* Gradient Background matching your reference image */
        .stApp {
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            color: #ffffff;
        }
        
        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: rgba(22, 27, 34, 0.95) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }

        /* Metrics Styling */
        [data-testid="stMetricValue"] {
            color: #a855f7; /* Purple accent from your image */
            font-weight: bold;
        }

        /* Glowing Buttons */
        .stButton>button {
            border-radius: 25px;
            background-color: #6366f1;
            color: white;
            font-weight: bold;
            border: none;
            transition: 0.3s;
        }
        .stButton>button:hover {
            box-shadow: 0 0 20px rgba(99, 102, 241, 0.8);
            background-color: #4f46e5;
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    apply_custom_design()

    # Initialize Session State for Page Navigation
    if "step" not in st.session_state:
        st.session_state.step = "upload" 
    if "rag" not in st.session_state:
        st.session_state.rag = None
    if "chunk_count" not in st.session_state:
        st.session_state.chunk_count = 0
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ---------------------------------------------------------
    # PAGE 1: UPLOAD & SETUP
    # ---------------------------------------------------------
    if st.session_state.step == "upload":
        st.title("📂 Knowledge Base Setup")
        st.write("Select your source to begin indexing your intelligence.")

        source_type = st.radio(
            "Choose your source:",
            ["Documents (PDF/TXT/CSV)", "Web URL", "Paste Text", "Wikipedia"],
            horizontal=True
        )

        processor = DocumentProcessor()
        all_chunks = []
        build_trigger = False

        # Container for Input Area
        with st.container():
            if source_type == "Documents (PDF/TXT/CSV)":
                files = st.file_uploader("Upload files", type=["pdf", "txt", "csv"], accept_multiple_files=True)
                if st.button("Build Knowledge Base") and files:
                    with st.spinner("Indexing documents..."):
                        for f in files:
                            all_chunks.extend(processor.process_uploaded_file(f))
                        build_trigger = True

            elif source_type == "Web URL":
                url_input = st.text_input("Enter website URL:")
                if st.button("Index Website") and url_input:
                    with st.spinner("Scraping..."):
                        all_chunks = processor.process_url(url_input)
                        build_trigger = True

            elif source_type == "Paste Text":
                raw_text = st.text_area("Paste text here:", height=300)
                if st.button("Index Text") and raw_text.strip():
                    all_chunks = processor.process_raw_text(raw_text)
                    build_trigger = True

            elif source_type == "Wikipedia":
                wiki_query = st.text_input("Enter topic:")
                if st.button("Fetch Wikipedia") and wiki_query:
                    all_chunks = processor.process_wikipedia(wiki_query)
                    build_trigger = True

        if build_trigger:
            if not all_chunks:
                st.error("Extraction failed. Check the source and try again.")
            else:
                vs = VectorStore()
                vs.create_vectorstore(all_chunks)
                gb = GraphBuilder(vs.get_retriever(), Config.get_llm())
                gb.build()

                st.session_state.rag = gb
                st.session_state.chunk_count = len(all_chunks)
                st.session_state.step = "chat" # Automatically move to Dashboard
                st.rerun()

    # ---------------------------------------------------------
    # PAGE 2: KNOWLEDGE DASHBOARD
    # ---------------------------------------------------------
    elif st.session_state.step == "chat":
        with st.sidebar:
            st.title("⚙️ Control Center")
            st.metric("Intelligence Chunks", st.session_state.chunk_count)
            if st.button("← Back to Upload"):
                st.session_state.step = "upload"
                st.rerun()
            st.divider()
            st.info("System Ready: Query the knowledge base below.")

        st.title("🤖 Knowledge Dashboard")

        # History Display
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat Input
        if prompt := st.chat_input("Ask a question based on your data..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    response = st.session_state.rag.run(prompt)
                    answer = response["answer"]
                    st.markdown(answer)
                    
                    with st.expander("View Reference Sources"):
                        for i, doc in enumerate(response.get("retrieved_docs", [])):
                            st.caption(f"Source {i+1}: {doc.page_content[:300]}...")
            
            st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()
