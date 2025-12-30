import os
import tempfile
import streamlit as st
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    WebBaseLoader,
    WikipediaLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound
)


class DocumentProcessor:
    def __init__(self, chunk_size=500, chunk_overlap=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # -------------------- COMMON SPLITTER --------------------
    def split_documents(self, docs):
        if not docs:
            return []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        return splitter.split_documents(docs)

    # -------------------- FILE UPLOAD --------------------
    def process_uploaded_file(self, uploaded_file):
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=Path(uploaded_file.name).suffix
        ) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            if uploaded_file.name.lower().endswith(".pdf"):
                loader = PyPDFLoader(tmp_path)
            elif uploaded_file.name.lower().endswith(".csv"):
                loader = CSVLoader(tmp_path)
            else:
                loader = TextLoader(tmp_path, encoding="utf-8")

            docs = loader.load()
            return self.split_documents(docs)

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            return []

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # -------------------- WEB URL --------------------
    def process_url(self, url):
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            return self.split_documents(docs)

        except Exception as e:
            st.error(f"Error loading URL: {str(e)}")
            return []

    # -------------------- YOUTUBE (CLOUD SAFE) --------------------
    def process_youtube(self, url):
        """
        Uses youtube-transcript-api instead of YoutubeLoader.
        This approach is more reliable on cloud platforms,
        but access may still be restricted by YouTube.
        """
        try:
            # Handle youtu.be and watch?v= formats
            if "youtu.be" in url:
                video_id = url.split("/")[-1].split("?")[0]
            else:
                video_id = url.split("v=")[-1].split("&")[0]

            transcript = YouTubeTranscriptApi.get_transcript(
                video_id,
                languages=["en"]
            )

            text = " ".join(chunk["text"] for chunk in transcript)

            docs = [
                Document(
                    page_content=text,
                    metadata={
                        "source": url,
                        "type": "youtube"
                    }
                )
            ]

            return self.split_documents(docs)

        except TranscriptsDisabled:
            st.error("This video does not have captions enabled.")
            return []

        except NoTranscriptFound:
            st.error("No English transcript was found for this video.")
            return []

        except Exception:
            st.error(
                "Unable to retrieve the YouTube transcript from this environment.\n\n"
                "Suggested actions:\n"
                "- Try a different video\n"
                "- Use a video with manually created captions\n"
                "- Paste or upload the transcript text manually"
            )
            return []

    # -------------------- WIKIPEDIA --------------------
    def process_wikipedia(self, query):
        """
        Accepts:
        - A topic name (e.g. 'Java Virtual Machine')
        - A full Wikipedia article URL
        """
        try:
            # Handle full Wikipedia URLs
            if "wikipedia.org/wiki/" in query:
                query = query.split("/wiki/")[-1].replace("_", " ")

            loader = WikipediaLoader(
                query=query,
                load_max_docs=1
            )

            docs = loader.load()
            return self.split_documents(docs)

        except Exception as e:
            st.error(f"Error loading Wikipedia content: {str(e)}")
            return []
