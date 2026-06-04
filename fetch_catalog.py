"""Step 0: download the latest deliberation catalogs from Rennes open data.

These JSON lists are what download_pdfs.py reads to learn which deliberations
exist. Re-fetching them is how new deliberations enter the pipeline.
"""
import json
import os
import urllib.request

CATALOGS = [
    {
        "url": "https://data.rennesmetropole.fr/api/explore/v2.1/catalog/datasets/deliberations-villerennes-2021/exports/json?lang=fr&timezone=Europe%2FBerlin",
        "output": "data/deliberations-villerennes-2021.json",
    },
    {
        "url": "https://data.rennesmetropole.fr/api/explore/v2.1/catalog/datasets/deliberations-rennes-metropole-2021-copie/exports/json?lang=fr&timezone=Europe%2FBerlin",
        "output": "data/deliberations-rennes-metropole-2021-copie.json",
    },
]

REQUIRED_KEYS = {"delib_id", "delib_url", "delib_date", "delib_matiere_nom", "delib_objet"}


def fetch(url, output):
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = resp.read()

    # Validate before replacing: a truncated download or a changed schema must not
    # clobber a working catalog, since every downstream step depends on it.
    records = json.loads(data)
    if not records:
        raise ValueError(f"{url} returned an empty list")
    missing = REQUIRED_KEYS - set(records[0])
    if missing:
        raise ValueError(f"{url} missing required keys: {missing}")

    tmp = output + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, output)
    print(f"{output}: {len(records)} deliberations")


def main():
    os.makedirs("data", exist_ok=True)
    for c in CATALOGS:
        fetch(c["url"], c["output"])


if __name__ == "__main__":
    main()
