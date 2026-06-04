import json
import os

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from config import E5_MODEL_PATH

# The model lives on disk; never reach out to HuggingFace (which is sometimes
# blocked on this machine and would just hang the run). Caller can override.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

CHUNKS = "data/chunks.json"
OUTPUT = "data/embeddings.npy"
# Sidecar that records, in order, which chunk_id each embedding row holds (plus
# the model that produced them). It lets re-runs embed ONLY the new chunks and
# append their rows, instead of recomputing all ~30k every time.
META = "data/embeddings.meta.json"

# The embeddings are locked to this model: the app must load the SAME model to
# embed incoming questions, or nearest-neighbour search compares unrelated spaces.
MODEL_NAME = E5_MODEL_PATH
BATCH_SIZE = 64


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def embed_chunks(model, chunks):
    # e5 requires a "passage: " prefix on documents (and "query: " on questions).
    texts = [f"passage: {c['text']}" for c in chunks]
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,  # unit length → dot product == cosine similarity
        convert_to_numpy=True,
    )
    # Store as float16 to roughly halve the index size; negligible quality loss.
    return embeddings.astype(np.float16)


def load_existing(current_ids):
    """Return (embeddings, n_reusable) for the rows we can keep as-is.

    We can reuse the existing index only if it was made by the same model and its
    chunk_ids are still a prefix of the current chunks (i.e. chunk.py only
    appended). Anything else (reordered, removed, different model) → re-embed all.
    """
    if not (os.path.exists(OUTPUT) and os.path.exists(META)):
        return None, 0
    with open(META) as f:
        meta = json.load(f)
    if meta.get("model") != MODEL_NAME:
        print("Model changed since last run — re-embedding everything.")
        return None, 0
    prev_ids = meta.get("chunk_ids", [])
    prev = np.load(OUTPUT)
    if len(prev) != len(prev_ids):
        print("Index/sidecar row count mismatch — re-embedding everything.")
        return None, 0
    if current_ids[: len(prev_ids)] != prev_ids:
        print("Existing chunks were reordered/removed — re-embedding everything.")
        return None, 0
    return prev, len(prev_ids)


def main():
    with open(CHUNKS) as f:
        chunks = json.load(f)
    current_ids = [c["chunk_id"] for c in chunks]

    prev, n_reusable = load_existing(current_ids)
    new_chunks = chunks[n_reusable:]

    if not new_chunks:
        print(f"Index already current: {len(chunks)} chunks, nothing to embed.")
        return

    device = pick_device()
    print(f"Loading {MODEL_NAME} on {device}...")
    model = SentenceTransformer(MODEL_NAME, device=device)

    print(f"Embedding {len(new_chunks)} new chunks ({n_reusable} reused)...")
    new_emb = embed_chunks(model, new_chunks)

    embeddings = new_emb if prev is None else np.vstack([prev, new_emb])
    np.save(OUTPUT, embeddings)
    with open(META, "w", encoding="utf-8") as f:
        json.dump(
            {"model": MODEL_NAME, "dim": int(embeddings.shape[1]), "chunk_ids": current_ids},
            f,
        )

    print(f"\nSaved {OUTPUT}: shape {embeddings.shape}, dtype {embeddings.dtype}")
    print(f"Size on disk: {embeddings.nbytes / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
