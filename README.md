# 📚 SmartDocs AI — Intelligent Document Q&A Platform

> Upload PDFs, ask questions in plain English, and get instant cited answers powered by OpenAI and ChromaDB.

---

## Architecture

```
PDF Upload
    │
    ▼
pdf_processor.py      ← PyMuPDF (primary) + pdfplumber (fallback)
    │
    ▼
text_cleaner.py       ← whitespace, header/footer removal, unicode normalisation
    │
    ▼
text_chunker.py       ← token-based or sentence-based chunking with overlap
    │
    ▼
embeddings.py         ← OpenAI text-embedding-ada-002 + persistent disk cache
    │
    ▼
vector_db.py          ← ChromaDB (local persistent store)
    │
    ▼
search_engine.py      ← cosine similarity search + reranking
    │
    ▼
qa_engine.py          ← GPT-3.5-turbo with streaming + citation enforcement
    │
    ▼
app.py                ← Streamlit UI (dark mode, PDF viewer, export, sessions)
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/your-username/smartdocs-ai.git
cd smartdocs-ai
pip install -r requirements.txt
```

### 2. Configure your OpenAI key

```bash
cp .env.template .env
# Open .env and paste your key from https://platform.openai.com/api-keys
```

### 3. Run

```bash
streamlit run frontend/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Features

| Feature | Details |
|---|---|
| **Multi-PDF upload** | Up to 10 MB per file, batch parallel processing |
| **Dual extraction** | PyMuPDF → pdfplumber fallback |
| **Smart chunking** | Token-based or sentence-based, dynamic sizing |
| **Persistent cache** | Embeddings cached to disk — no repeat API calls |
| **Deduplication** | Re-uploading the same file skips re-ingestion |
| **Streaming answers** | Token-by-token streaming via GPT-3.5-turbo |
| **Inline citations** | Every fact cited `[Source: file.pdf, Page: N]` |
| **Document deletion** | Remove any indexed doc and its chunks via UI |
| **Dark mode** | Full dark/light theme toggle |
| **Session export** | Download conversation as `.txt`, `.md`, or `.json` |
| **Health check** | Verifies DB, API key, and disk space |

---

## Project Structure

```
SmartDocs AI/
├── config.py                  # All constants in one place
├── frontend/
│   └── app.py                 # Streamlit UI
├── backend/
│   ├── pdf_processor.py
│   ├── text_cleaner.py
│   ├── text_chunker.py
│   ├── embeddings.py
│   ├── vector_db.py
│   ├── ingestion_pipeline.py
│   ├── batch_processor.py
│   ├── search_engine.py
│   ├── qa_engine.py
│   └── session_manager.py
├── utils/
│   ├── error_handler.py
│   └── logging_config.py
├── tests/
│   ├── test_text_cleaner.py
│   ├── test_text_chunker.py
│   ├── test_embeddings.py
│   ├── test_vector_db.py
│   ├── test_qa_engine.py
│   └── test_ingestion_pipeline.py
├── data/
│   ├── chroma_db/             # ChromaDB storage (auto-created)
│   ├── sessions/              # Session JSON files (auto-created)
│   └── embedding_cache.json   # Persistent embedding cache (auto-created)
├── logs/
│   └── app.log                # Unified application log
├── .env                       # Your API key (never commit)
├── .env.template              # Safe template to commit
├── requirements.txt
└── pytest.ini
```

---

## Running Tests

```bash
pytest
```

All tests mock OpenAI API calls — no credits consumed during testing.

---

## Configuration

All tunable constants are in `config.py`:

```python
CHAT_MODEL            = "gpt-3.5-turbo"
EMBEDDING_MODEL       = "text-embedding-ada-002"
DEFAULT_CHUNK_SIZE    = 1000
DEFAULT_SIMILARITY_THRESHOLD = 0.7
SESSION_TIMEOUT_HOURS = 24
```

---

## Tech Stack

- **Frontend**: Streamlit
- **LLM**: OpenAI GPT-3.5-turbo (streaming)
- **Embeddings**: OpenAI text-embedding-ada-002
- **Vector DB**: ChromaDB (local persistent)
- **PDF**: PyMuPDF + pdfplumber
- **Tokenisation**: tiktoken + NLTK