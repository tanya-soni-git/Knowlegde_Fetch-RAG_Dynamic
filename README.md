# 📂 Knowlegde_Fetch-RAG_Dynamic

### *Dynamic Multi-Source Agentic RAG*

A professional, agentic Retrieval-Augmented Generation (RAG) application built with **LangGraph**, **LangChain**, and **Streamlit**. This tool enables users to build a custom knowledge base from multiple sources and interact with it through a futuristic, dark-themed dashboard.

---

## ✨ Features

* **Multi-Source Ingestion:** Support for PDF, TXT, CSV, Web URLs, and Wikipedia topics.
* **Agentic Workflow:** Powered by a `StateGraph` for managed document retrieval and answer generation.
* **Modern UI:** Professional dashboard with a page-by-page setup flow and dark-mode CSS styling.
* **Vector Search:** Utilizes **FAISS** for high-performance similarity searches.
* **Real-time Metrics:** Displays exact "Knowledge Chunks" created during the indexing process.

---

## 🛠️ Tech Stack

| Component | Technology |
| :--- | :--- |
| **Frontend** | Streamlit |
| **Orchestration** | LangGraph & LangChain |
| **LLM Provider** | Groq (Llama-3.1-8b) |
| **Vector Database** | FAISS |
| **Embeddings** | HuggingFace (all-MiniLM-L6-v2) |

---

## 🚀 Getting Started

### Prerequisites
* Python 3.11+
* A **Groq API Key** (Set as `XAI_API_KEY` in your `.env` or Streamlit secrets)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/tanya-soni-git/knowlegde_fetch-rag_dynamic.git
   cd knowlegde_fetch-rag_dynamic

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt

3. **Environment Setup: Create a .env file in the root directory:**
   ```bash
   XAI_API_KEY=your_groq_api_key_here

4. **Launch Application:**
   ```bash
   streamlit run streamlit_app.py

### 🏗️ Project Structure

```text
src/
├── config/             # Configuration & LLM setup
├── document_ingestion/ # File/URL processing logic
├── graph_builder/      # LangGraph workflow orchestration
├── node/               # RAG functional nodes (Retrieve/Generate)
├── state/              # TypedDict state definitions
└── vectorstore/        # FAISS & Embedding management
data/                   # Local data storage
streamlit_app.py        # Main UI & Application entry point
requirements.txt        # Project dependencies
README.md               # Project documentation
```
```markdown
### 🚀 Live Demo
https://knowlegdefetch-ragdynamic-ai.streamlit.app
```
### App ScreenShots 
<img width="1905" height="1003" alt="image" src="https://github.com/user-attachments/assets/93b9bb44-e05e-43a9-a220-ac3c69c9806f" />
<img width="1905" height="1022" alt="image" src="https://github.com/user-attachments/assets/b475e142-b7c1-4fea-a05e-be0754b9884f" />
<img width="1900" height="939" alt="image" src="https://github.com/user-attachments/assets/82a3c953-11da-4b82-aa8d-090c8804c753" />
<img width="1910" height="906" alt="image" src="https://github.com/user-attachments/assets/e6470336-cf2d-44cc-8dc8-25c5bfe21eec" />
<img width="1874" height="956" alt="image" src="https://github.com/user-attachments/assets/a108535f-9ddf-4846-a300-a348c8d843fd" />
<img width="1908" height="1005" alt="image" src="https://github.com/user-attachments/assets/f07b1cd2-9597-48c8-83d5-a5545c6c15cc" />
<img width="1911" height="1007" alt="image" src="https://github.com/user-attachments/assets/c3500ab1-fe13-44de-912c-0070fd228676" />
<img width="1898" height="975" alt="image" src="https://github.com/user-attachments/assets/ff89ed92-ea96-4455-9155-902e8adee12a" />

### Key Improvements Made:
* **Visual Hierarchy:** Added horizontal rules (`---`) to separate major sections.
* **Information Density:** Used a **Table** for the Tech Stack to make it scannable.
* **Project Tree:** Added a visual directory structure so users understand where the code lives.
* **Consistent Styling:** Standardized bolding and icon usage to match the "Business Pitch Deck" theme of your UI.

