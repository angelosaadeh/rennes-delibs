import json
import os
import re
import fitz

DATASETS = [
    {
        "json": "data/deliberations-villerennes-2021.json",
        "pdf_dir": "data/pdfs-ville",
        "output": "data/extracted-ville.json",
    },
    {
        "json": "data/deliberations-rennes-metropole-2021-copie.json",
        "pdf_dir": "data/pdfs-metropole",
        "output": "data/extracted-metropole.json",
    },
]


def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text() for page in doc).strip()


# Phrases that mark the end of the attendee block / start of the decision body.
# Needed because the two formats differ: the Bureau/Ville format ends the block
# with "Le quorum…", while the full Conseil métropolitain has no quorum line and
# runs straight into "… est nommé secrétaire de séance" / "Le Conseil constate".
# Without these terminators the regex runs off the end and swallows the whole text.
ATTENDEE_TERMINATORS = (
    r"Ont donn[ée]|Absents?|Excus[ée]s?|Le quorum|"
    r"est nomm[ée]+\s+secr[ée]taire|Le Conseil constate|Le Bureau constate|"
    r"La s[ée]ance est lev[ée]e|Participaient? également|"
    r"Envoy[ée] en préfecture|Reçu en préfecture|\*\s|$"
)


def parse_attendees(text):
    attendees = {}
    heads = {
        "presents": r"Pr[ée]sents?\s*:\s*(.+?)",
        "procuration": r"Ont donn[ée] procuration\s*:\s*(.+?)",
        "absents": r"Absents?(?:/Excus[ée]s?)?\s*:\s*(.+?)",
    }
    for key, head in heads.items():
        pattern = head + r"(?=" + ATTENDEE_TERMINATORS + r")"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            value = re.sub(r"\s+", " ", match.group(1)).strip().rstrip(".,;")
            if value:
                attendees[key] = value
    return attendees


def process_dataset(json_path, pdf_dir, output_path):
    with open(json_path) as f:
        records = json.load(f)

    # Incremental: keep whatever we already extracted and only process
    # deliberations we haven't seen, appending them. Extraction is cheap-ish but
    # re-parsing thousands of PDFs on every catalog refresh is pure waste.
    if os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
    else:
        results = []
    done_ids = {r["delib_id"] for r in results}

    missing = new = 0

    for r in records:
        if r["delib_id"] in done_ids:
            continue
        pdf_path = os.path.join(pdf_dir, f"{r['delib_id']}.pdf")
        if not os.path.exists(pdf_path):
            missing += 1
            continue

        try:
            text = extract_text(pdf_path)
        except Exception as e:
            print(f"  FAIL {r['delib_id']}: {e}")
            continue

        results.append({
            "delib_id": r["delib_id"],
            "delib_date": r["delib_date"],
            "delib_matiere_nom": r["delib_matiere_nom"],
            "delib_objet": r["delib_objet"],
            "attendees": parse_attendees(text),
            "text": text,
        })
        # The Rennes catalog occasionally lists the same delib_id twice; mark it
        # done immediately so we don't extract (and later chunk) it twice.
        done_ids.add(r["delib_id"])
        new += 1

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"{output_path}: {new} newly extracted, {len(results)} total, {missing} PDFs missing")


if __name__ == "__main__":
    for ds in DATASETS:
        process_dataset(ds["json"], ds["pdf_dir"], ds["output"])
