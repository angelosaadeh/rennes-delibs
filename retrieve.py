"""Retrieval half of the RAG pipeline: question -> most relevant chunks.

Kept separate from the LLaMA answer step so retrieval can be tested on its own
(it loads in a couple of seconds; the 6.6 GB LLaMA does not).
"""
import gzip
import json
import os
import urllib.request

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from config import E5_MODEL_PATH, RETRIEVAL_MAX_K, RETRIEVAL_MIN_K

# Stay offline only when the embedding model is already on disk. On a hosted
# Space, E5_MODEL_PATH is a Hub id (e.g. "intfloat/multilingual-e5-base") that
# must be downloaded, so forcing offline there would break startup.
if os.path.isdir(E5_MODEL_PATH):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

# MUST be the same model that built data/embeddings.npy — the question vector and
# the chunk vectors have to live in the same space for cosine search to mean
# anything. embeddings.meta.json records which model that was.
MODEL_NAME = E5_MODEL_PATH
CHUNKS = "data/chunks.json.gz"
EMBEDDINGS = "data/embeddings.npy"
META = "data/embeddings.meta.json"

# If the index isn't present locally (e.g. a fresh hosted Space), fetch it from
# the GitHub repo instead of committing 44 MB of binaries to the Space.
INDEX_BASE_URL = os.environ.get(
    "INDEX_BASE_URL",
    "https://raw.githubusercontent.com/angelosaadeh/rennes-delibs/main/data",
)


def _ensure_index(path):
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    url = f"{INDEX_BASE_URL}/{os.path.basename(path)}"
    print(f"Téléchargement de l'index : {url}")
    urllib.request.urlretrieve(url, path)

# Dynamic selection: instead of a fixed top-k, look at the top MAX_K candidates
# and cut where the scores drop off the most (an "elbow"), no earlier than MIN_K.
# A broad question whose relevance fades slowly keeps more chunks; a narrow one
# with a sharp drop after the first few keeps fewer. e5 cosine scores sit in a
# compressed band, so a relative gap adapts far better than an absolute cutoff.
# The ceiling is backend-aware (config): higher for Groq's large context window.
MIN_K = RETRIEVAL_MIN_K
MAX_K = RETRIEVAL_MAX_K


def _model_name(path):
    """Last path component, so /a/b/e5, models/e5 and org/e5 all compare equal."""
    return os.path.basename(path.rstrip("/"))


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Retriever:
    def __init__(self):
        for path in (CHUNKS, EMBEDDINGS, META):
            _ensure_index(path)
        with gzip.open(CHUNKS, "rt", encoding="utf-8") as f:
            self.chunks = json.load(f)
        # Index is fp16 on disk; promote to fp32 for an accurate dot product.
        self.embeddings = np.load(EMBEDDINGS).astype(np.float32)
        if len(self.chunks) != len(self.embeddings):
            raise SystemExit(
                f"Index out of sync: {len(self.chunks)} chunks vs "
                f"{len(self.embeddings)} embeddings. Re-run embed.py."
            )
        # Guard against querying with a different model than built the index.
        # Compare by name (last path component) rather than full path, so the
        # same model loaded from a different location — or from the HF Hub id
        # "intfloat/multilingual-e5-base" — still matches.
        if os.path.exists(META):
            with open(META) as f:
                model = json.load(f).get("model")
            if model and _model_name(model) != _model_name(MODEL_NAME):
                raise SystemExit(
                    f"Index was built with {model}, but retrieve uses {MODEL_NAME}."
                )
        self.model = SentenceTransformer(MODEL_NAME, device=pick_device())

    def search(self, question, k=None, min_k=MIN_K, max_k=MAX_K):
        """Return [(score, chunk)] for the matches.

        By default the count is *dynamic*: among the top `max_k` candidates, cut
        at the biggest score drop (no earlier than `min_k`). Pass an explicit `k`
        to force a fixed number instead.
        """
        # e5 expects "query: " on questions (documents used "passage: ").
        q = self.model.encode(
            [f"query: {question}"],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0].astype(np.float32)
        # Both sides are unit-normalized, so the dot product is cosine similarity.
        scores = self.embeddings @ q
        order = np.argsort(-scores)  # 29k items — full sort is cheap and exact

        if k is not None:
            selected = order[: min(k, len(order))]
        else:
            cand = order[: min(max_k, len(order))]
            cs = scores[cand]
            if len(cand) > min_k:
                gaps = cs[:-1] - cs[1:]  # gaps[j] = score drop right after rank j
                # Largest drop at rank >= min_k-1 → keep everything up to it.
                cut = (min_k - 1) + int(np.argmax(gaps[min_k - 1:]))
                selected = cand[: cut + 1]
            else:
                selected = cand
        return [(float(scores[i]), self.chunks[i]) for i in selected]


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]).strip() or "Quelles décisions concernant les pistes cyclables ?"
    r = Retriever()
    print(f"Q: {question}\n")
    for score, c in r.search(question):
        print(f"[{score:.3f}] {c['type']:9} {c['delib_date']}  {c['delib_objet'][:72]}")
