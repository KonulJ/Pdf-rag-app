# PDF RAG App

A local Retrieval-Augmented Generation (RAG) application that lets you chat with your PDF documents — no cloud, no GPU required.

## Tech Stack

| Layer | Tool |
|---|---|
| Framework | LlamaIndex |
| LLM | Ollama + `phi3:mini` |
| Embeddings | Ollama + `nomic-embed-text` |
| Vector Store | ChromaDB (local, persistent) |
| PDF Parser | PyMuPDF |
| UI | Streamlit |

## Project Structure

```
pdf-rag-app/
├── app.py              # Streamlit UI
├── rag/
│   ├── config.py       # Model names, chunk sizes, paths
│   ├── indexer.py      # PDF ingestion and ChromaDB indexing
│   └── retriever.py    # Query engine and source extraction
├── data/pdfs/          # Uploaded PDFs (git-ignored)
├── storage/chroma/     # Persisted vector index (git-ignored)
└── requirements.txt
```

## Prerequisites

### 1. Install Ollama
Download from [ollama.com](https://ollama.com) and install it.

### 2. Pull the required models
```bash
ollama pull phi3:mini
ollama pull nomic-embed-text
```

### 3. Make sure Ollama is running
```bash
ollama serve
```

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/your-username/pdf-rag-app.git
cd pdf-rag-app

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Usage

1. Upload one or more PDFs using the sidebar
2. Wait for indexing to complete (one-time per document)
3. Type your question in the chat box
4. The app retrieves the most relevant chunks and generates an answer
5. Expand **View sources** under each answer to see which pages were used

## Configuration

Edit [rag/config.py](rag/config.py) to change:

| Setting | Default | Description |
|---|---|---|
| `LLM_MODEL` | `phi3:mini` | Any model available in Ollama |
| `EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `CHUNK_SIZE` | `512` | Tokens per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `TOP_K` | `5` | Number of chunks retrieved per query |

### Alternative models (no GPU needed)

| Model | Command | Notes |
|---|---|---|
| Llama 3.2 3B | `ollama pull llama3.2:3b` | Slightly larger, very capable |
| Mistral 7B Q4 | `ollama pull mistral` | Slower on CPU, higher quality |
| Phi-3 Mini | `ollama pull phi3:mini` | Default — fast on CPU |

## How It Works

```
PDF Upload
    ↓
PyMuPDF extracts text page by page
    ↓
SentenceSplitter chunks the text (512 tokens, 50 overlap)
    ↓
nomic-embed-text embeds each chunk → ChromaDB stores vectors
    ↓
User asks a question
    ↓
Question is embedded → top-5 similar chunks are retrieved
    ↓
phi3:mini generates an answer from the retrieved chunks
    ↓
Answer + source citations shown in Streamlit
```

## License

MIT
