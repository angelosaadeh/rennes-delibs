"""Download the two local models into the paths from config.py (skips if present).

    python download_models.py

Targets config.E5_MODEL_PATH and config.LLAMA_MODEL_PATH, so set those env vars
(or edit config.py) first if you want them somewhere other than the defaults.
The LLaMA file is ~6.6 GB, so this can take a while on the first run.
"""
import os
import urllib.request

from config import E5_MODEL_PATH, LLAMA_MODEL_PATH, LLM_BACKEND

E5_BASE = "https://huggingface.co/intfloat/multilingual-e5-base/resolve/main"
# Everything sentence-transformers needs to load the model from a local folder.
E5_FILES = [
    "config.json",
    "sentence_bert_config.json",
    "modules.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
    "1_Pooling/config.json",
    "model.safetensors",
]
LLAMA_URL = (
    "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/"
    "resolve/main/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf"
)


def download(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  skip (exists): {dest}")
        return
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = dest + ".tmp"
    print(f"  downloading {os.path.basename(dest)} ...")
    # Temp-then-rename so an interrupted download never leaves a partial file
    # that would be wrongly skipped next time.
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dest)


def main():
    # e5 is always needed: it embeds the user's question at query time.
    print(f"e5 embedding model -> {E5_MODEL_PATH}")
    for f in E5_FILES:
        download(f"{E5_BASE}/{f}", os.path.join(E5_MODEL_PATH, *f.split("/")))

    # The big LLaMA GGUF is only needed for the local backend; Groq runs it hosted.
    if LLM_BACKEND == "local":
        print(f"LLaMA GGUF -> {LLAMA_MODEL_PATH}")
        download(LLAMA_URL, LLAMA_MODEL_PATH)
    else:
        print(f"LLM_BACKEND={LLM_BACKEND}: skipping the LLaMA download (hosted).")
    print("Done.")


if __name__ == "__main__":
    main()
