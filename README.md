# PDF RAG App

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.10%2B-blueviolet)
![ChromaDB](https://img.shields.io/badge/ChromaDB-vector--store-green)
![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-grey)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A fully local Retrieval-Augmented Generation (RAG) application — upload PDFs, ask questions, get answers with source-page citations. No cloud, no GPU, no API keys required.

---

## Tech Stack

| Layer | Tool |
| --- | --- |
| Framework | LlamaIndex |
| LLM | Ollama + `phi3:mini` |
| Embeddings | Ollama + `nomic-embed-text` |
| Vector Store | ChromaDB (local, persistent) |
| PDF Parser | PyMuPDF |
| UI | Streamlit |

---

## Project Structure

```text
pdf-rag-app/
├── app.py              # Streamlit UI — upload, chat, source citations
├── rag/
│   ├── config.py       # Model names, chunk sizes, paths
│   ├── indexer.py      # PDF ingestion and ChromaDB indexing
│   └── retriever.py    # Streaming query engine and source extraction
├── data/pdfs/          # Uploaded PDFs (git-ignored)
├── storage/chroma/     # Persisted vector index (git-ignored)
└── requirements.txt
```

---

## Prerequisites

### 1. Install Ollama

Download from [ollama.com](https://ollama.com) and install it.

### 2. Pull the required models

```bash
ollama pull phi3:mini
ollama pull nomic-embed-text
```

### 3. Start Ollama

```bash
ollama serve
```

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/KonulJ/Pdf-rag-app.git
cd pdf-rag-app

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Usage

1. Upload one or more PDFs using the sidebar
2. Wait for indexing to complete (one-time per document)
3. Type your question in the chat box
4. The app retrieves the most relevant chunks and streams the answer
5. Expand **View sources** under each answer to see which pages were used

---

## How It Works

```text
PDF Upload
    ↓
PyMuPDF extracts text page by page
    ↓
SentenceSplitter chunks text (512 tokens, 50 overlap)
    ↓
nomic-embed-text embeds each chunk → stored in ChromaDB
    ↓
User asks a question
    ↓
Question embedded → top-5 similar chunks retrieved
    ↓
phi3:mini streams an answer grounded in retrieved chunks
    ↓
Answer streams token-by-token + source citations shown
```

---

## Key Concepts Demonstrated

| Concept | Implementation |
| --- | --- |
| RAG pipeline | LlamaIndex `VectorStoreIndex` + `SentenceSplitter` |
| Local vector store | ChromaDB persistent client — no cloud dependency |
| Local LLM inference | Ollama — runs entirely on CPU |
| Streaming generation | `query_engine(streaming=True)` + `st.write_stream` |
| Source citations | `response.source_nodes` — file, page, score, text preview |
| Modular design | Separate `config`, `indexer`, `retriever` modules |

---

## Configuration

Edit [rag/config.py](rag/config.py) to change models or chunking behaviour:

| Setting | Default | Description |
| --- | --- | --- |
| `LLM_MODEL` | `phi3:mini` | Any model available in Ollama |
| `EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `CHUNK_SIZE` | `512` | Tokens per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `TOP_K` | `5` | Number of chunks retrieved per query |

### Alternative models (no GPU needed)

| Model | Pull command | Notes |
| --- | --- | --- |
| Llama 3.2 3B | `ollama pull llama3.2:3b` | Slightly larger, very capable |
| Mistral 7B Q4 | `ollama pull mistral` | Slower on CPU, higher quality |
| Phi-3 Mini | `ollama pull phi3:mini` | Default — fast on CPU |

---

## License

MIT

---

*Built by [Konul Jafarova](https://github.com/KonulJ)*
