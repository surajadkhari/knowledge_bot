"""
KnowledgeBot — Document Ingestion
Reads local files and URLs, chunks them, embeds with sentence-transformers,
and stores in ChromaDB for later retrieval.
"""

from pathlib import Path

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import pypdf

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────

CHUNK_SIZE = 500       # characters per chunk
CHUNK_OVERLAP = 50     # overlap between chunks
EMBED_MODEL = "all-MiniLM-L6-v2"
DATA_DIR = Path(__file__).parent / "data"
CHROMA_DIR = Path(__file__).parent / "chroma_db"

# URLs to ingest (add your company wiki, docs, etc.)
URLS = [
    # "https://example.com/company-policy",
]


# ── Helpers ─────────────────────────────────────────────────────────────────

def chunk_text(text: str, source: str) -> list[dict]:
    """Split text into overlapping chunks with metadata."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end]
        chunks.append({"text": chunk, "source": source})
        if end == len(text):
            break
        start = end - CHUNK_OVERLAP
    return chunks


def read_file(path: Path) -> str | None:
    """Read a file, return text content or None if unsupported."""
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            reader = pypdf.PdfReader(path)
            return "\n".join(
                page.extract_text() or "" for page in reader.pages
            )
        elif suffix in (".txt", ".md"):
            return path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"  ⚠ Skipping {path.name}: {e}")
    return None


def fetch_url(url: str) -> str | None:
    """Fetch and extract text from a URL."""
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "KnowledgeBot/1.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)
    except Exception as e:
        print(f"  ⚠ Skipping {url}: {e}")
    return None


# ── Custom ChromaDB embedding function ──────────────────────────────────────

class STEmbeddingFunction:
    """Wraps sentence-transformers for ChromaDB."""
    def __init__(self, model_name: str = EMBED_MODEL):
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


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("🔧 Loading embedding model...")
    emb_fn = STEmbeddingFunction()

    print("📂 Setting up ChromaDB...")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))

    # Recreate collection each ingest for simplicity
    try:
        client.delete_collection("knowledgebot")
    except Exception:
        pass

    collection = client.create_collection(
        name="knowledgebot",
        embedding_function=emb_fn,
    )

    all_chunks = []
    doc_count = 0

    # ── Local files ─────────────────────────────────────────────────────
    if DATA_DIR.exists():
        for path in sorted(DATA_DIR.iterdir()):
            if path.is_file():
                text = read_file(path)
                if text:
                    chunks = chunk_text(text, source=path.name)
                    all_chunks.extend(chunks)
                    doc_count += 1
                    print(f"  ✓ {path.name} → {len(chunks)} chunks")

    # ── URLs ────────────────────────────────────────────────────────────
    for url in URLS:
        text = fetch_url(url)
        if text:
            chunks = chunk_text(text, source=url)
            all_chunks.extend(chunks)
            doc_count += 1
            print(f"  ✓ {url} → {len(chunks)} chunks")

    if not all_chunks:
        print("❌ No documents found. Add files to data/ or URLs to the URLS list.")
        return

    # ── Store in ChromaDB ───────────────────────────────────────────────
    print(f"\n📥 Storing {len(all_chunks)} chunks from {doc_count} documents...")
    for i, chunk in enumerate(all_chunks):
        collection.add(
            ids=[str(i)],
            documents=[chunk["text"]],
            metadatas=[{"source": chunk["source"]}],
        )

    print(f"✅ Done! {len(all_chunks)} chunks indexed in ChromaDB.")
    print("   Run: streamlit run app.py")


if __name__ == "__main__":
    main()
