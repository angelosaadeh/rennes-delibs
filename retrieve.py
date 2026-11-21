"""Retrieval half of the RAG pipeline: question -> most relevant chunks.

Kept separate from the LLaMA answer step so retrieval can be tested on its own
(it loads in a couple of seconds; the 6.6 GB LLaMA does not).
"""
import gzip
import json
import os
import re
import urllib.request

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from config import E5_MODEL_PATH, RETRIEVAL_MAX_K

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

# Plain top-k: return the MAX_K highest-scoring chunks. An earlier "elbow" cut
# (stop at the biggest score gap) fired far too early here — e5 cosine scores sit
# in a compressed band, so the largest gap lands in the first few ranks and only a
# handful of chunks came back regardless of MAX_K. Top-k is predictable and
# actually uses the ceiling, which is backend-aware (config): higher for Groq.
MAX_K = RETRIEVAL_MAX_K

# Attendee chunks carry the same topical [objet | date] header as the decision
# body, so for a topical query ("subventions santé") a name roster scores as high
# as the decision that actually answers it — and, being short and uniform, a whole
# block of rosters can monopolise the top ranks and crowd the decisions out
# entirely. We therefore drop attendee chunks from retrieval UNLESS the question is
# itself about attendance, detected by these words.
ATTENDANCE_RE = re.compile(
    r"\b(pr[ée]sent|absent|procuration|quorum|si[ée]g|assist|excus|"
    r"qui (?:[ée]tait|sont|a particip)|vot[ae])",
    re.IGNORECASE,
)


def _is_attendance_query(question):
    return bool(ATTENDANCE_RE.search(question))


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

        # delib_id -> indices of its decision chunks, kept in document order so a
        # retrieved chunk can pull in the rest of its deliberation (see search()).
        self.decision_siblings = {}
        for i, c in enumerate(self.chunks):
            if c["type"] == "decision":
                self.decision_siblings.setdefault(c["delib_id"], []).append(i)

    def search(self, question, k=None, max_k=MAX_K, expand_radius=3,
               include_attendees=None):
        """Return [(score, chunk)] for the best matches, highest cosine first.

        `max_k` (or an explicit `k`) is a budget on the number of chunks RETURNED,
        not on the number of distinct matches scored. Two corrections to plain
        top-k are applied so the budget is spent on text that can actually answer
        the question:

        * Attendee chunks are excluded unless the question is about attendance
          (`include_attendees`, auto-detected when None). They share the decision's
          topical header and would otherwise dominate the ranking with name lists.
        * Sibling expansion (`expand_radius`): when a decision chunk matches, its
          neighbouring chunks from the SAME deliberation (the `expand_radius`
          chunks on either side, in document order) are pulled in with it. A
          deliberation's amounts/beneficiaries often sit a chunk or two away from
          the chunk carrying the topical header, in a table that — being mostly
          names and numbers — scores far below it; expansion brings it along. The
          window is bounded, so a single very long deliberation (some budget
          documents run to hundreds of chunks) cannot swallow the whole budget.
          Set `expand_radius=0` to disable and recover plain top-k.
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

        if include_attendees is None:
            include_attendees = _is_attendance_query(question)
        budget = k if k is not None else max_k
        # An attendance question wants the compact rosters, not deliberation
        # bodies — expanding decision windows would spend the budget before the
        # roster chunks (which rank a little lower) are reached.
        if include_attendees:
            expand_radius = 0

        results, picked = [], set()
        for i in order:
            if len(picked) >= budget:
                break
            c = self.chunks[i]
            if i in picked or (c["type"] == "attendees" and not include_attendees):
                continue
            # Window of neighbours from the same deliberation, in document order.
            if c["type"] == "decision" and expand_radius:
                sib = self.decision_siblings[c["delib_id"]]
                pos = sib.index(i)
                group = sib[max(0, pos - expand_radius): pos + expand_radius + 1]
            else:
                group = [i]
            # Add what fits; never overflow the budget (truncate a window rather
            # than let one deliberation monopolise the context).
            for j in group:
                if len(picked) >= budget:
                    break
                if j not in picked:
                    picked.add(j)
                    results.append((float(scores[i]), self.chunks[j]))
        return results


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]).strip() or "Quelles décisions concernant les pistes cyclables ?"
    r = Retriever()
    print(f"Q: {question}\n")
    for score, c in r.search(question):
        print(f"[{score:.3f}] {c['type']:9} {c['delib_date']}  {c['delib_objet'][:72]}")
