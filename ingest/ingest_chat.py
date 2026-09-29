"""
ingest_chat.py — Chat History Vectoriser for Hey-Neo

Converts a single chat turn (user query + Neo's response) into dense
vector embeddings and upserts them into the Qdrant `chat_history`
collection.

Usage (called from ui.py after each completed turn):
    from ingest.ingest_chat import ingest_chat_turn
    ingest_chat_turn(user_input="what GPU do I have?", neo_response="You have an RTX 4070.")

Collection schema:
    - size  : 1024  (mxbai-embed-large output dimension)
    - metric: Cosine
    - payload fields:
        role      : "user" | "neo"
        text      : the raw text of this chunk
        turn_id   : shared UUID for the user+neo pair in one conversation turn
        chunk_idx : int, position of this chunk within the role's text
        source    : always "chat_history"
"""

import uuid
import re

import ollama
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

COLLECTION_NAME = "chat_history"
EMBED_MODEL     = "mxbai-embed-large:latest"
VECTOR_SIZE     = 1024

# Chunk config — keep chunks small so retrieval stays focused
MAX_WORDS = 150
OVERLAP   = 30

# ──────────────────────────────────────────────────────────────────────────────
# Chunking
# ──────────────────────────────────────────────────────────────────────────────

_noise = re.compile(r'[:<>@,\[\]{}#]')


def _split_into_chunks(text: str, max_words: int = MAX_WORDS, overlap: int = OVERLAP) -> list[str]:
    """
    Split *text* into overlapping word-windows.

    Mirrors the split_into_chunks() logic from similarity_ingest.py so
    embedding quality stays consistent across all collections.
    """
    words = [
        w for w in text.split()
        if not w.startswith('/')
        and len(w) <= 35
        and not w.startswith("#")
        and not w.startswith("_")
        and not _noise.search(w)
        and any(c.isalpha() for c in w)
    ]

    if not words:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += max_words - overlap
    return chunks


# ──────────────────────────────────────────────────────────────────────────────
# Qdrant helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_client() -> QdrantClient:
    return QdrantClient(url="http://localhost:6333")


def _ensure_collection(qdrant: QdrantClient) -> None:
    """Create the chat_history collection if it doesn't already exist."""
    if not qdrant.collection_exists(COLLECTION_NAME):
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Embedding
# ──────────────────────────────────────────────────────────────────────────────

def _embed(text: str) -> list[float]:
    """Return the dense embedding vector for *text* via Ollama."""
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return response["embedding"]


# ──────────────────────────────────────────────────────────────────────────────
# Core public function
# ──────────────────────────────────────────────────────────────────────────────

def ingest_chat_turn(user_input: str, neo_response: str) -> None:
    """
    Vectorise and upsert one chat turn (user + Neo) into Qdrant.

    Each side of the conversation is chunked independently, embedded,
    and stored with a shared `turn_id` so they can be linked at
    retrieval time.

    Args:
        user_input  : The raw query the user typed.
        neo_response: The final answer Neo produced (accumulated string
                      from the Phase-B stream in ui.py).
    """
    if not user_input.strip() and not neo_response.strip():
        return

    turn_id = str(uuid.uuid4())
    qdrant  = _get_client()
    _ensure_collection(qdrant)

    points: list[PointStruct] = []

    for role, text in [("user", user_input), ("neo", neo_response)]:
        if not text.strip():
            continue

        chunks = _split_into_chunks(text)

        # If the text is short and chunker drops everything (all noise), fall
        # back to the raw stripped text so we never silently lose a turn.
        if not chunks:
            chunks = [text.strip()]

        for idx, chunk in enumerate(chunks):
            print(f"[chat ingest] role={role} chunk={idx} words={len(chunk.split())}")
            vector = _embed(chunk)
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "role":      role,
                        "text":      chunk,
                        "turn_id":  turn_id,
                        "chunk_idx": idx,
                        "source":    "chat_history",
                    },
                )
            )

    if points:
        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )
        print(f"[chat ingest] upserted {len(points)} point(s) for turn {turn_id}")