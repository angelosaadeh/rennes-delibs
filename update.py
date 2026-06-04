"""Run the full pipeline end to end, incrementally.

Each step skips work already done, so after the first build this is fast: only
NEW deliberations get downloaded, extracted, chunked, and embedded.

    ~/myenv/bin/python3 update.py

It starts by refreshing the catalog JSONs from Rennes open data, so newly
published deliberations are picked up automatically.
"""
import subprocess
import sys

STEPS = [
    ("Refresh catalogs", "fetch_catalog.py"),
    ("Download new PDFs", "download_pdfs.py"),
    ("Extract text", "extract_text.py"),
    ("Chunk", "chunk.py"),
    ("Embed", "embed.py"),
]


def main():
    for label, script in STEPS:
        print(f"\n=== {label} ({script}) ===", flush=True)
        subprocess.run([sys.executable, script], check=True)
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
