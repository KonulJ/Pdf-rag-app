import os
import streamlit as st

from rag.config import PDF_DIR
from rag.indexer import clear_index, get_indexed_files, index_pdf, load_index
from rag.retriever import extract_sources, query_index

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF RAG",
    page_icon="📄",
    layout="wide",
)

# ── Session state defaults ─────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📂 Documents")
    st.caption("Upload PDFs to ask questions about them.")

    uploaded = st.file_uploader("Choose a PDF", type=["pdf"], label_visibility="collapsed")

    if uploaded:
        os.makedirs(PDF_DIR, exist_ok=True)
        save_path = os.path.join(PDF_DIR, uploaded.name)

        already_indexed = uploaded.name in get_indexed_files()
        if already_indexed:
            st.info(f"**{uploaded.name}** is already indexed.")
        else:
            with st.spinner(f"Indexing {uploaded.name}..."):
                try:
                    with open(save_path, "wb") as f:
                        f.write(uploaded.getbuffer())
                    index_pdf(save_path)
                    st.success(f"Indexed: **{uploaded.name}**")
                except Exception as exc:
                    st.error(f"Failed to index: {exc}")

    st.divider()

    # List indexed files
    indexed_files = get_indexed_files()
    if indexed_files:
        st.subheader("Indexed files")
        for name in indexed_files:
            st.markdown(f"- {name}")
    else:
        st.info("No documents indexed yet.")

    st.divider()

    # Clear index button
    if indexed_files:
        if st.button("Clear all documents", type="secondary", use_container_width=True):
            clear_index()
            st.session_state.messages = []
            st.success("Index cleared.")
            st.rerun()

    # Model info
    st.divider()
    st.caption("**Models (via Ollama)**")
    st.caption("LLM: phi3:mini")
    st.caption("Embeddings: nomic-embed-text")


# ── Main area ──────────────────────────────────────────────────────────────────
st.title("📄 PDF RAG — Ask Your Documents")
st.caption("Upload a PDF in the sidebar, then ask any question about its content.")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("View sources"):
                for src in msg["sources"]:
                    score_label = f" | Score: {src['score']}" if src["score"] is not None else ""
                    st.markdown(f"**{src['file']}** — Page {src['page']}{score_label}")
                    st.caption(src["text"])
                    st.divider()

# Chat input
if prompt := st.chat_input("Ask a question about your PDFs..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        index = load_index()

        if index is None:
            answer = "Please upload and index at least one PDF first."
            sources: list[dict] = []
            st.markdown(answer)
        else:
            with st.spinner("Searching documents and generating answer..."):
                try:
                    response = query_index(index, prompt)
                    answer = str(response)
                    sources = extract_sources(response)
                except Exception as exc:
                    answer = f"Something went wrong: {exc}"
                    sources = []

            st.markdown(answer)

            if sources:
                with st.expander("View sources"):
                    for src in sources:
                        score_label = f" | Score: {src['score']}" if src["score"] is not None else ""
                        st.markdown(f"**{src['file']}** — Page {src['page']}{score_label}")
                        st.caption(src["text"])
                        st.divider()

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
