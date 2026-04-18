from __future__ import annotations

import os

from llama_index.core import VectorStoreIndex

from rag.config import TOP_K


def query_index(index: VectorStoreIndex, question: str):
    """Return a StreamingResponse — source_nodes populated immediately, response_gen streams the answer."""
    query_engine = index.as_query_engine(
        similarity_top_k=TOP_K,
        streaming=True,
    )
    return query_engine.query(question)


def extract_sources(response) -> list[dict]:
    """
    Pull citation metadata out of a LlamaIndex Response.

    Returns a list of dicts with keys:
        file   – filename of the source PDF
        page   – page label (if available)
        score  – cosine-similarity score (if available)
        text   – first 300 characters of the chunk
    """
    sources = []
    for node_with_score in response.source_nodes:
        node = node_with_score.node
        meta = node.metadata or {}
        sources.append(
            {
                "file":  os.path.basename(meta.get("file_name") or meta.get("source") or "unknown"),
                "page":  meta.get("page_label") or meta.get("page") or "N/A",
                "score": round(node_with_score.score, 3) if node_with_score.score is not None else None,
                "text":  node.text[:300].strip() + ("..." if len(node.text) > 300 else ""),
            }
        )
    return sources
