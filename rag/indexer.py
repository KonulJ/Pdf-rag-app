import os
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.readers.file import PyMuPDFReader

from rag.config import (
    OLLAMA_BASE_URL, EMBED_MODEL, LLM_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP,
    CHROMA_PATH, COLLECTION_NAME,
)


def _setup_settings() -> None:
    """Configure global LlamaIndex settings (models, chunk sizes)."""
    Settings.embed_model = OllamaEmbedding(
        model_name=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    Settings.llm = Ollama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        request_timeout=180.0,
    )
    Settings.chunk_size = CHUNK_SIZE
    Settings.chunk_overlap = CHUNK_OVERLAP


def _get_chroma_collection():
    """Return a persistent ChromaDB client and collection."""
    os.makedirs(CHROMA_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)
    return client, collection


def load_index() -> VectorStoreIndex | None:
    """
    Load an existing index from ChromaDB.
    Returns None if no documents have been indexed yet.
    """
    _setup_settings()
    client, collection = _get_chroma_collection()

    if collection.count() == 0:
        return None

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context
    )


def index_pdf(file_path: str) -> VectorStoreIndex:
    """
    Parse a PDF, embed its chunks, and persist them in ChromaDB.
    Re-uses any existing documents already in the collection.
    """
    _setup_settings()
    client, collection = _get_chroma_collection()

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    reader = PyMuPDFReader()
    documents = reader.load(file_path)

    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[splitter],
        show_progress=True,
    )
    return index


def get_indexed_files() -> list[str]:
    """Return a deduplicated list of file names already in the index."""
    _, collection = _get_chroma_collection()
    if collection.count() == 0:
        return []

    results = collection.get(include=["metadatas"])
    files: set[str] = set()
    for meta in results["metadatas"]:
        if meta:
            name = meta.get("file_name") or meta.get("source") or ""
            if name:
                files.add(os.path.basename(name))
    return sorted(files)


def clear_index() -> None:
    """Delete the entire ChromaDB collection (removes all indexed documents)."""
    client, _ = _get_chroma_collection()
    client.delete_collection(COLLECTION_NAME)
