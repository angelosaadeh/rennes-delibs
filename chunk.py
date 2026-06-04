import gzip
import json
import os
import re

INPUTS = [
    {"file": "data/extracted-ville.json", "source": "ville"},
    {"file": "data/extracted-metropole.json", "source": "metropole"},
]
# Gzipped so the prebuilt index can be committed to the repo (~12 MB vs ~63 MB).
OUTPUT = "data/chunks.json.gz"

# Target sizes in characters. ~1800 chars ≈ 450 tokens, so several chunks fit
# comfortably inside LLaMA 3.1 8B's context (n_ctx=4096) at retrieval time.
CHUNK_SIZE = 1800
CHUNK_OVERLAP = 200

# Repetitive boilerplate stamped on every page footer — pure noise for retrieval.
FOOTER_PATTERNS = [
    r"Envoy[ée] en préfecture le.*",
    r"Reçu en préfecture le.*",
    r"Publié le.*",
    r"Affiché le.*",
    r"ID\s*:\s*035-[\w-]+",
]
FOOTER_RE = re.compile("|".join(FOOTER_PATTERNS), re.IGNORECASE)

# Markers for where the actual decision content begins, after the attendee
# preamble. We keep the earliest match. If none is found (older 2021 format),
# we leave the body untouched rather than risk deleting real content.
CONTENT_MARKERS = [
    r"EXPOS[ÉE]",
    r"Après en avoir délibéré",
    r"DÉLIBÈRE",
    r"D[ÉE]CIDE",
    r"Vu le ",
    r"Vu la ",
    r"Considérant",
]
CONTENT_RE = re.compile("|".join(CONTENT_MARKERS))


def strip_preamble(text):
    """Drop the attendee/quorum preamble — it lives in its own chunk already."""
    match = CONTENT_RE.search(text)
    if match and match.start() > 0:
        return text[match.start():]
    return text


def clean_body(text):
    text = FOOTER_RE.sub("", text)
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        # Drop bare page numbers and empty lines left behind by footer removal.
        if not stripped or stripped.isdigit():
            continue
        lines.append(stripped)
    return "\n".join(lines)


def split_text(text):
    """Paragraph-aware splitting into ~CHUNK_SIZE pieces with overlap."""
    paragraphs = [p for p in text.split("\n") if p]
    chunks, current = [], ""

    for para in paragraphs:
        if len(current) + len(para) + 1 <= CHUNK_SIZE:
            current = f"{current}\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
            # Start next chunk with a tail of the previous one for context overlap.
            tail = current[-CHUNK_OVERLAP:] if current else ""
            current = f"{tail}\n{para}" if tail else para
            # A single oversized paragraph still needs hard splitting.
            while len(current) > CHUNK_SIZE:
                chunks.append(current[:CHUNK_SIZE])
                current = current[CHUNK_SIZE - CHUNK_OVERLAP:]
    if current:
        chunks.append(current)
    return chunks


def format_attendees(attendees):
    parts = []
    if attendees.get("presents"):
        parts.append(f"Présents : {attendees['presents']}")
    if attendees.get("procuration"):
        parts.append(f"Ont donné procuration : {attendees['procuration']}")
    if attendees.get("absents"):
        parts.append(f"Absents/Excusés : {attendees['absents']}")
    return "\n".join(parts)


def make_chunks(record, source):
    # A short context header is prepended to every chunk so an isolated chunk
    # still says which deliberation it belongs to — improves retrieval quality.
    header = f"[{record['delib_objet']} | {record['delib_date']}]"
    base = {
        "delib_id": record["delib_id"],
        "delib_date": record["delib_date"],
        "delib_matiere_nom": record["delib_matiere_nom"],
        "delib_objet": record["delib_objet"],
        "source": source,
    }
    out = []

    # Dedicated path for attendees: one self-contained chunk, so "who attended?"
    # queries hit it directly without drowning the decision chunks.
    attendees_text = format_attendees(record.get("attendees", {}))
    if attendees_text:
        out.append({
            **base,
            "chunk_id": f"{record['delib_id']}#attendees",
            "type": "attendees",
            "text": f"{header}\n\n{attendees_text}",
        })

    # Decision body, cleaned, preamble stripped, then chunked.
    body = strip_preamble(clean_body(record["text"]))
    for i, chunk in enumerate(split_text(body)):
        out.append({
            **base,
            "chunk_id": f"{record['delib_id']}#decision-{i}",
            "type": "decision",
            "text": f"{header}\n\n{chunk}",
        })
    return out


def main():
    # Incremental + append-only: load the existing chunks, then add chunks only
    # for deliberations not already present. Appending at the END is what keeps
    # embeddings.npy aligned — row i must keep pointing at the same chunk i, so we
    # must never reorder or insert in the middle.
    if os.path.exists(OUTPUT):
        with gzip.open(OUTPUT, "rt", encoding="utf-8") as f:
            all_chunks = json.load(f)
    else:
        all_chunks = []
    done_ids = {c["delib_id"] for c in all_chunks}

    for inp in INPUTS:
        with open(inp["file"]) as f:
            records = json.load(f)
        n_before = len(all_chunks)
        added = 0
        for r in records:
            if r["delib_id"] in done_ids:
                continue
            all_chunks.extend(make_chunks(r, inp["source"]))
            done_ids.add(r["delib_id"])
            added += 1
        print(f"{inp['file']}: {added} new deliberations → {len(all_chunks) - n_before} new chunks")

    with gzip.open(OUTPUT, "wt", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    n_att = sum(1 for c in all_chunks if c["type"] == "attendees")
    n_dec = sum(1 for c in all_chunks if c["type"] == "decision")
    print(f"\n{OUTPUT}: {len(all_chunks)} chunks total ({n_att} attendees, {n_dec} decision)")


if __name__ == "__main__":
    main()
