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
        "json": "data/deliberations-rennes-metropole-2021.json",
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

    results = []
    missing = 0

    for r in records:
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

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"{output_path}: {len(results)} extracted, {missing} PDFs missing")


if __name__ == "__main__":
    for ds in DATASETS:
        process_dataset(ds["json"], ds["pdf_dir"], ds["output"])
