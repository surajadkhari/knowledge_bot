"""
KnowledgeBot — RAG Chat Interface
Streamlit app that retrieves relevant document chunks and answers
questions using OpenAI's GPT model.
"""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────

EMBED_MODEL = "all-MiniLM-L6-v2"
CHROMA_DIR = Path(__file__).parent / "chroma_db"
TOP_K = 5

# ── Page setup ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="KnowledgeBot", page_icon="📚")
st.title("📚 KnowledgeBot")
st.caption("Ask questions about your company's knowledge base.")

# ── Init (cached across reruns) ────────────────────────────────────────────


@st.cache_resource
def init_chroma():
    """Load ChromaDB collection and embedding model."""
    if not CHROMA_DIR.exists():
        return None, None

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

    class STEmbeddingFunction:
        def __init__(self, model_name=EMBED_MODEL):
            self.model = SentenceTransformer(model_name)
            self._name = model_name

        def __call__(self, input: list[str]) -> list[list[float]]:
            return self._embed(input)

        def embed_query(self, input: list[str]) -> list[list[float]]:
            return self._embed(input)

        def _embed(self, input: list[str]) -> list[list[float]]:
            return self.model.encode(input, normalize_embeddings=True).tolist()

        def name(self) -> str:
            return self._name

        @staticmethod
        def build_from_config(config):
            return STEmbeddingFunction(model_name=config.get("model_name", EMBED_MODEL))

        @staticmethod
        def get_config():
            return {"model_name": EMBED_MODEL}

    emb_fn = STEmbeddingFunction()
    collection = client.get_collection(name="knowledgebot", embedding_function=emb_fn)
    return collection, emb_fn.model


@st.cache_resource
def init_llm():
    """Initialize LLM client (OpenAI-compatible, e.g. DeepSeek)."""
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
    if not api_key:
        return None, None
    return OpenAI(api_key=api_key, base_url=base_url), os.getenv(
        "LLM_MODEL", "deepseek-chat"
    )


collection, embed_model = init_chroma()
client, llm_model = init_llm()

# ── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")

    if not os.getenv("LLM_API_KEY"):
        st.error("Set LLM_API_KEY in your .env file")
    elif collection is None:
        st.warning("No index found. Run: python ingest.py")
    else:
        st.success(f"✅ Index ready ({collection.count()} chunks)")

    st.divider()
    st.caption("Add files to `data/` and run `python ingest.py` to update.")

# ── Chat ────────────────────────────────────────────────────────────────────

# Init chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    if not client:
        st.error("LLM API key not configured. Check your .env file.")
        st.stop()
    if collection is None:
        st.error("No documents indexed. Run: python ingest.py")
        st.stop()

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ── Retrieve ────────────────────────────────────────────────────────
    with st.spinner("Searching knowledge base..."):
        results = collection.query(query_texts=[prompt], n_results=TOP_K)
        sources = results["metadatas"][0]
        chunks = results["documents"][0]

    # ── Generate ────────────────────────────────────────────────────────
    context = "\n\n---\n\n".join(
        f"[Source: {s['source']}]\n{c}" for c, s in zip(chunks, sources)
    )

    system_prompt = (
        "Always mentioned responed that I am Mr Bin.You are a helpful knowledge base assistant. Answer the user's question "
        "using ONLY the provided context below. If the context doesn't contain "
        "the answer, say 'I couldn't find that in the knowledge base.' "
        "Cite your sources when possible.\n\n"
        f"CONTEXT:\n{context}"
    )

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )
        response = st.write_stream(stream)

    st.session_state.messages.append({"role": "assistant", "content": response})

    # ── Show sources in sidebar ─────────────────────────────────────────
    with st.sidebar:
        st.divider()
        st.subheader("🔍 Sources Used")
        seen = set()
        for s in sources:
            if s["source"] not in seen:
                seen.add(s["source"])
                st.caption(f"• {s['source']}")
