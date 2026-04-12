# ── Ollama settings ────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL     = "nomic-embed-text"   # ollama pull nomic-embed-text
LLM_MODEL       = "phi3:mini"          # ollama pull phi3:mini

# ── Chunking ───────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 512
CHUNK_OVERLAP = 50

# ── Retrieval ──────────────────────────────────────────────────────────────────
TOP_K = 5

# ── Storage paths ──────────────────────────────────────────────────────────────
PDF_DIR         = "data/pdfs"
CHROMA_PATH     = "storage/chroma"
COLLECTION_NAME = "pdf_rag"
