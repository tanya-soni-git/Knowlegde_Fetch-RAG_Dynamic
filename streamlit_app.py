import streamlit as st
from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder

st.set_page_config(page_title="Dynamic Agentic RAG", page_icon="📂", layout="wide")

def main():
    st.title("📂 Dynamic Multi-Source RAG")
    
    if 'rag' not in st.session_state:
        st.session_state.rag = None

    # Sidebar UI for selecting the section
    with st.sidebar:
        st.header("Input Section")
        source_type = st.radio(
            "Choose your source:",
            ["Documents (PDF/TXT/CSV)", "Web URL", "YouTube Video", "Wikipedia"]
        )
        
        processor = DocumentProcessor()
        all_chunks = []
        build_trigger = False

        # Section 1: Files
        if source_type == "Documents (PDF/TXT/CSV)":
            uploaded_files = st.file_uploader(
                "Upload files", 
                type=['pdf', 'txt', 'csv'], 
                accept_multiple_files=True
            )
            if st.button("Build Knowledge Base") and uploaded_files:
                with st.spinner("Indexing files..."):
                    for f in uploaded_files:
                        chunks = processor.process_uploaded_file(f)
                        all_chunks.extend(chunks)
                    build_trigger = True

        # Section 2: URL
        elif source_type == "Web URL":
            url_input = st.text_input("Enter Website URL:")
            if st.button("Index Website") and url_input:
                with st.spinner("Scraping website..."):
                    all_chunks = processor.process_url(url_input)
                    build_trigger = True

        # Section 3: YouTube
        elif source_type == "YouTube Video":
            yt_input = st.text_input("Enter YouTube Video URL:")
            if st.button("Index Transcript") and yt_input:
                with st.spinner("Fetching transcript..."):
                    all_chunks = processor.process_youtube(yt_input)
                    build_trigger = True

        # Section 4: Wikipedia
        elif source_type == "Wikipedia":
            wiki_query = st.text_input("Enter Search Topic:")
            if st.button("Index Wikipedia") and wiki_query:
                with st.spinner("Searching Wikipedia..."):
                    all_chunks = processor.process_wikipedia(wiki_query)
                    build_trigger = True

        # Common Build Logic
        if build_trigger:
            if not all_chunks:
                st.error("No content could be extracted. Please check your input.")
            else:
                vs = VectorStore()
                vs.create_vectorstore(all_chunks)
                gb = GraphBuilder(vs.get_retriever(), Config.get_llm())
                gb.build()
                st.session_state.rag = gb
                st.success(f"Success! {len(all_chunks)} chunks indexed.")

    # Main Chat Interface
    if st.session_state.rag:
        user_input = st.text_input("Ask a question about the provided context:")
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
        st.info("👈 Please use the sidebar to upload documents or links to begin.")

if __name__ == "__main__":
    main()
