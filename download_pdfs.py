import json
import os
import ssl
import time
import urllib.request

DATASETS = [
    {
        "json": "data/deliberations-villerennes-2021.json",
        "output_dir": "data/pdfs-ville",
    },
    {
        "json": "data/deliberations-rennes-metropole-2021-copie.json",
        "output_dir": "data/pdfs-metropole",
    },
]

# The Mégalis server presents a certificate with a hostname mismatch, so TLS
# verification is disabled on purpose. Acceptable here since this is public data.
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


def download_dataset(json_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path) as f:
        records = [r for r in json.load(f) if r.get("delib_url")]

    print(f"\n{json_path}: {len(records)} PDFs to download → {output_dir}")
    ok = fail = 0

    for i, r in enumerate(records):
        dest = os.path.join(output_dir, f"{r['delib_id']}.pdf")
        if os.path.exists(dest):
            ok += 1
            continue
        try:
            with urllib.request.urlopen(r["delib_url"], context=ctx, timeout=15) as resp:
                data = resp.read()
            # Write to a temp file then rename, so an interrupted download never
            # leaves a partial .pdf that would be wrongly skipped on the next run.
            tmp = dest + ".tmp"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, dest)
            ok += 1
        except Exception as e:
            fail += 1
            print(f"  FAIL {r['delib_id']}: {e}")
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(records)} done...")
        time.sleep(0.2)

    total_mb = sum(
        os.path.getsize(os.path.join(output_dir, f))
        for f in os.listdir(output_dir)
        if f.endswith(".pdf")
    ) / 1024 / 1024

    print(f"  Done. Success: {ok}, Failed: {fail}, Total size: {total_mb:.1f} MB")


if __name__ == "__main__":
    for ds in DATASETS:
        download_dataset(ds["json"], ds["output_dir"])
