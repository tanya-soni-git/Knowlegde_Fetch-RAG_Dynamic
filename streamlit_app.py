import streamlit as st
from langchain_core.documents import Document

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
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
