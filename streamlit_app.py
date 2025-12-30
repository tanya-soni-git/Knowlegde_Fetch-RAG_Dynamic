import streamlit as st
from langchain_core.documents import Document

from src.config.config import Config
from src.document_ingestiimport streamlit as st
from langchain_core.documents import Document

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
    """Injects custom CSS to achieve the dark, neon-accented 'Pitch Deck' look."""
    st.markdown("""
        <style>
        /* Gradient Background */
        .stApp {
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            color: #ffffff;
        }
        
        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: rgba(22, 27, 34, 0.95) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }

        /* Glassmorphism containers for inputs */
        div[data-testid="stVerticalBlock"] > div:has(div.stButton) {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 25px;
            backdrop-filter: blur(10px);
        }

        /* Metrics Styling */
        [data-testid="stMetricValue"] {
            color: #a855f7; /* Purple accent */
            font-weight: bold;
        }

        /* Chat Input Styling */
        .stChatInputContainer {
            padding-bottom: 20px;
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

    # Initialize Session State
    if "step" not in st.session_state:
        st.session_state.step = "upload" # Initial page
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
        st.title("📂 Build Knowledge Base")
        st.write("Welcome! Select your data source to begin indexing.")

        # Modern horizontal source selection
        source_type = st.radio(
            "Choose your input source:",
            ["Documents (PDF/TXT/CSV)", "Web URL", "Paste Text", "Wikipedia"],
            horizontal=True
        )

        processor = DocumentProcessor()
        all_chunks = []
        build_trigger = False

        # Input areas wrapped in a styled container
        with st.container():
            if source_type == "Documents (PDF/TXT/CSV)":
                files = st.file_uploader("Upload files", type=["pdf", "txt", "csv"], accept_multiple_files=True)
                if st.button("Build Knowledge Base") and files:
                    with st.spinner("Processing files..."):
                        for f in files:
                            all_chunks.extend(processor.process_uploaded_file(f))
                        build_trigger = True

            elif source_type == "Web URL":
                url_input = st.text_input("Enter URL:")
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
                if st.button("Fetch Wiki Content") and wiki_query:
                    all_chunks = processor.process_wikipedia(wiki_query)
                    build_trigger = True

        # Knowledge Base Generation Logic
        if build_trigger:
            if not all_chunks:
                st.error("Extraction failed. Please check the source content.")
            else:
                vs = VectorStore()
                vs.create_vectorstore(all_chunks)
                gb = GraphBuilder(vs.get_retriever(), Config.get_llm())
                gb.build()

                st.session_state.rag = gb
                st.session_state.chunk_count = len(all_chunks)
                st.session_state.step = "chat" # Move to Page 2
                st.rerun()

    # ---------------------------------------------------------
    # PAGE 2: KNOWLEDGE DASHBOARD (CHAT)
    # ---------------------------------------------------------
    elif st.session_state.step == "chat":
        # Sidebar for status and navigation
        with st.sidebar:
            st.header("📊 Overview")
            st.metric("Chunks Created", st.session_state.chunk_count)
            st.success("Knowledge Base Active")
            
            if st.button("← Back to Upload"):
                st.session_state.step = "upload"
                st.session_state.messages = [] # Reset chat for new context
                st.rerun()
            
            st.divider()
            st.caption("Powered by Dynamic RAG Agent")

        st.title("🤖 AI Knowledge Assistant")

        # Display Chat History
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat Input Area
        if prompt := st.chat_input("Ask a question about your indexed data..."):
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate Assistant Answer
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.rag.run(prompt)
                    answer = response["answer"]
                    st.markdown(answer)
                    
                    # Show sources inside an expander in the chat bubble
                    with st.expander("View Reference Sources"):
                        for i, doc in enumerate(response.get("retrieved_docs", [])):
                            st.info(f"Source {i+1}: {doc.page_content[:300]}...")
            
            st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()on.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder


st.set_page_config(
    page_title="Dynamic RAG",
    page_icon="📂",
    layout="wide"
)


def main():
    st.title("Dynamic Multi-Source RAG")

    if "rag" not in st.session_state:
        st.session_state.rag = None

    # -------------------- SIDEBAR --------------------
    with st.sidebar:
        st.header("Input Source")

        source_type = st.radio(
            "Choose your source:",
            [
                "Documents (PDF/TXT/CSV)",
                "Web URL",
                "Paste Text",
                "Wikipedia"
            ]
        )

        processor = DocumentProcessor()
        all_chunks = []
        build_trigger = False

        # -------------------- DOCUMENTS --------------------
        if source_type == "Documents (PDF/TXT/CSV)":
            uploaded_files = st.file_uploader(
                "Upload files",
                type=["pdf", "txt", "csv"],
                accept_multiple_files=True
            )

            if st.button("Build Knowledge Base") and uploaded_files:
                with st.spinner("Indexing documents..."):
                    for f in uploaded_files:
                        chunks = processor.process_uploaded_file(f)
                        all_chunks.extend(chunks)
                    build_trigger = True

        # -------------------- WEB URL --------------------
        elif source_type == "Web URL":
            url_input = st.text_input("Enter website URL:")

            if st.button("Index Website") and url_input:
                with st.spinner("Scraping website..."):
                    all_chunks = processor.process_url(url_input)
                    build_trigger = True

        # -------------------- PASTE TEXT --------------------
        elif source_type == "Paste Text":
            raw_text = st.text_area(
                "Paste text here:",
                height=280,
                placeholder="Paste articles, notes, transcripts, or any raw text..."
            )

            if st.button("Index Text") and raw_text.strip():
                with st.spinner("Indexing text..."):
                    all_chunks = processor.process_raw_text(raw_text)
                    build_trigger = True

        # -------------------- WIKIPEDIA --------------------
        elif source_type == "Wikipedia":
            wiki_query = st.text_input("Enter topic or Wikipedia URL:")

            if st.button("Index Wikipedia") and wiki_query:
                with st.spinner("Fetching Wikipedia content..."):
                    all_chunks = processor.process_wikipedia(wiki_query)
                    build_trigger = True

        # -------------------- BUILD VECTORSTORE --------------------
        if build_trigger:
            if not all_chunks:
                st.error("No content could be extracted. Please check your input.")
            else:
                vs = VectorStore()
                vs.create_vectorstore(all_chunks)

                gb = GraphBuilder(
                    vs.get_retriever(),
                    Config.get_llm()
                )
                gb.build()

                st.session_state.rag = gb
                st.success(f"Knowledge base built with {len(all_chunks)} chunks.")

    # -------------------- MAIN CHAT --------------------
    if st.session_state.rag:
        user_input = st.text_input("Ask a question based on the indexed content:")

        if user_input:
            with st.spinner("Generating answer..."):
                response = st.session_state.rag.run(user_input)

                st.markdown("### Answer")
                st.write(response["answer"])

                with st.expander("View Retrieved Sources"):
                    for i, doc in enumerate(response.get("retrieved_docs", [])):
                        st.markdown(f"Source {i + 1}")
                        st.info(doc.page_content)

    else:
        st.info("Use the sidebar to upload content and build the knowledge base.")


if __name__ == "__main__":
    main()
