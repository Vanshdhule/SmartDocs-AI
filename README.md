# 📚 SmartDocs AI — Intelligent Document Q&A Platform

> Upload PDFs, ask questions in plain English, and get instant cited answers —
> powered by LangChain, NVIDIA NIM dual LLM backends, and ChromaDB.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)
![LangChain](https://img.shields.io/badge/LangChain-0.2.16-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🚀 Quick Start

### 1. Clone and install
```bash
git clone https://github.com/your-username/smartdocs-ai.git
cd smartdocs-ai
python -m venv venv
# Windows:  venv\Scripts\activate
# Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure NVIDIA API keys
```bash
cp .env.template .env
# Edit .env and paste your free NVIDIA NIM keys
# Get them at: https://build.nvidia.com → Sign In → Get API Key
```

`.env` contents:
```
NVIDIA_CHAT_API_KEY=nvapi-your-chat-key-here
NVIDIA_EMBEDDING_API_KEY=nvapi-your-embedding-key-here
```

### 3. Run
```bash
streamlit run frontend/app.py
```
Open [http://localhost:8501](http://localhost:8501)

---

## 🏗️ Architecture

```
PDF Upload
    │
    ├─► pdf_processor.py       PyMuPDF (primary) → pdfplumber (fallback)
    │
    ├─► text_cleaner.py        Permissive cleaning — preserves technical content
    │
    ├─► text_chunker.py        Token-based or sentence-based chunking with overlap
    │
    ├─► embeddings.py          nvidia/nv-embedqa-e5-v5 (1024-dim) + disk cache
    │
    ├─► vector_db.py           ChromaDB — workspace-namespaced collections
    │
    ├─► search_engine.py       Cosine similarity + reranking + 3-tier fallback
    │
    └─► qa_engine.py           LangChain LCEL | ChatOpenAI | Streaming | Memory
                                       │
                               frontend/app.py    Streamlit UI (dark theme)
```

---

## ✨ Features

| Feature | Details |
|---|---|
| **Workspace Isolation** | Each topic = separate ChromaDB collection. No cross-contamination between unrelated document sets. |
| **Multi-PDF per Workspace** | Upload multiple PDFs into one workspace for combined knowledge search. |
| **Dual LLM Backends** | Meta Llama 3.3 70B + NVIDIA Nemotron 70B — runtime-switchable, no restart needed. |
| **3-Tier Fault Tolerance** | Llama → Nemotron → raw chunks. User always gets a useful response. |
| **LangChain LCEL Pipeline** | `ChatPromptTemplate \| ChatOpenAI \| StrOutputParser` with `ConversationBufferWindowMemory`. |
| **Streaming Responses** | Token-by-token via LangChain `.stream()` with live cursor in UI. |
| **Persistent Embedding Cache** | JSON disk cache — embeddings survive restarts, no repeat API costs. |
| **Deduplication** | Skip or replace already-indexed documents, per workspace. |
| **AI Document Summaries** | Auto-generated 2-3 sentence summary per document on index. |
| **Inline Citations** | Every fact cited `[Source: file.pdf, Page: N]`. |
| **Document Deletion** | Remove any doc and its chunks from the UI. |
| **Workspace Diagnostics** | Real-time chunk count per document in sidebar panel. |
| **Session Export** | Download conversation as `.txt`, `.md`, or `.json`. |
| **Health Check** | Verifies ChromaDB, NVIDIA API keys, and disk space. |

---

## 🤖 Dual LLM Backends

| Model | API String | Best For |
|---|---|---|
| **Llama 3.3 70B** | `meta/llama-3.3-70b-instruct` | Fast, accurate general Q&A |
| **Nemotron 70B** | `nvidia/llama-3.1-nemotron-70b-instruct` | RLHF-tuned for grounded factual accuracy |

Both are free on [build.nvidia.com](https://build.nvidia.com) with an OpenAI-compatible API.

---

## 🗂️ Workspace System

```
ChromaDB/
├── smartdocs_default/          ← General workspace
├── smartdocs_ai_research/      ← AI Research papers
├── smartdocs_harry_potter/     ← Harry Potter series
└── smartdocs_legal_docs/       ← Legal documents
```

- **Isolated search** — questions only search within the active workspace
- **Combined knowledge** — multiple PDFs in one workspace = combined answers across all
- **Zero contamination** — Harry Potter never appears in AI Research answers

---

## ⚡ 3-Tier Fault-Tolerant Fallback

```
User Question
      │
      ▼
┌─────────────────────────────┐
│ Tier 1: Llama 3.3 70B       │  ← normal path
└─────────────────────────────┘
      │ fails (rate limit / timeout)
      ▼
┌─────────────────────────────┐
│ Tier 2: Nemotron 70B        │  ← auto-switch via LangChain
└─────────────────────────────┘
      │ fails
      ▼
┌─────────────────────────────┐
│ Tier 3: Raw chunk display   │  ← user always gets something
└─────────────────────────────┘
```

---

## 📁 Project Structure

```
SmartDocs AI/
├── config.py                      # All constants centralised
├── frontend/
│   └── app.py                     # Streamlit UI (dark theme, workspace UI)
├── backend/
│   ├── __init__.py
│   ├── pdf_processor.py           # PyMuPDF + pdfplumber fallback
│   ├── text_cleaner.py            # Permissive cleaning (preserves citations)
│   ├── text_chunker.py            # Token + sentence strategies
│   ├── embeddings.py              # NVIDIA embeddings + disk cache
│   ├── vector_db.py               # Workspace-aware ChromaDB wrapper
│   ├── ingestion_pipeline.py      # Full PDF→embed→store pipeline
│   ├── batch_processor.py         # Parallel ingestion (3 workers)
│   ├── search_engine.py           # RAG + 3-tier fallback
│   ├── qa_engine.py               # LangChain LCEL + streaming + memory
│   ├── session_manager.py         # Workspace + session persistence
│   └── openai_helper.py           # NVIDIA NIM connection helper
├── utils/
│   ├── __init__.py
│   ├── error_handler.py           # Validation + health check
│   └── logging_config.py          # Unified logging
├── tests/
│   ├── conftest.py                # fitz DLL pre-mock
│   ├── test_text_cleaner.py       # 22 tests — no API calls
│   ├── test_text_chunker.py       # 14 tests — no API calls
│   ├── test_embeddings.py         # 10 tests — mocked
│   ├── test_vector_db.py          # 18 tests — real ephemeral ChromaDB
│   ├── test_qa_engine.py          # 22 tests — LangChain mocked
│   ├── test_ingestion_pipeline.py # 10 tests — full pipeline mocked
│   ├── test_nvidia.py             # manual smoke test (live keys needed)
│   └── test_openai.py             # manual smoke test (live keys needed)
├── scripts/                       # Dev/debug tools (not part of CI)
│   ├── test_chunk_ids.py
│   ├── test_extraction.py
│   ├── test_pdf_to_embeddings.py
│   ├── test_search_engine.py
│   └── test_unique_chunk_ids.py
├── data/                          # Auto-created, gitignored
│   ├── chroma_db/                 # ChromaDB workspace collections
│   ├── sessions/                  # Session JSON files
│   └── embedding_cache.json       # Persistent embedding cache
├── logs/                          # Auto-created, gitignored
│   └── app.log
├── uploads/                       # PDFs saved here, gitignored
├── .env                           # Your NVIDIA keys — NEVER commit
├── .env.template                  # Safe template to commit
├── requirements.txt
├── pytest.ini
└── README.md
```

---

## 🧪 Running Tests

```bash
# Automated suite — no API keys needed
pytest tests/test_text_cleaner.py \
       tests/test_text_chunker.py \
       tests/test_embeddings.py \
       tests/test_vector_db.py \
       tests/test_qa_engine.py \
       tests/test_ingestion_pipeline.py -v

# Live API smoke tests (requires .env with NVIDIA keys)
python tests/test_nvidia.py
python tests/test_openai.py
```

---

## ⚙️ Configuration

All settings in `config.py`:

```python
# Dual LLM backends
PRIMARY_MODEL   = "meta/llama-3.3-70b-instruct"
SECONDARY_MODEL = "nvidia/llama-3.1-nemotron-70b-instruct"

# Embeddings
EMBEDDING_MODEL     = "nvidia/nv-embedqa-e5-v5"
EMBEDDING_DIMENSION = 1024

# Retrieval
DEFAULT_TOP_K                = 12
DEFAULT_SIMILARITY_THRESHOLD = 0.25

# Chunking
DEFAULT_CHUNK_SIZE    = 500
DEFAULT_CHUNK_OVERLAP = 100

# Workspace
WORKSPACE_COLLECTION_PREFIX = "smartdocs_"
DEFAULT_WORKSPACE_NAME      = "Default"

# Session
SESSION_TIMEOUT_HOURS = 24
AUTO_SAVE_EVERY       = 5
```

---

## 📋 Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit 1.35 (dark theme) |
| **LLM Orchestration** | LangChain 0.2.16 (LCEL, ConversationBufferWindowMemory, streaming) |
| **LLM Provider** | NVIDIA NIM (free tier, OpenAI-compatible) |
| **Embeddings** | nvidia/nv-embedqa-e5-v5 (1024-dim) |
| **Vector DB** | ChromaDB 0.5 (local persistent, workspace-namespaced) |
| **PDF Parsing** | PyMuPDF 1.24 + pdfplumber 0.11 |
| **Tokenisation** | tiktoken 0.7 + NLTK 3.8 |
| **Testing** | pytest 8.2 + pytest-mock 3.14 |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
