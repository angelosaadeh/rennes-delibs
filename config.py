"""Machine-dependent settings, in one place.

Everything here can be overridden with an environment variable (or a .env you
source), so nothing else in the codebase hardcodes a path tied to one computer.
On a new machine, either set these env vars or edit the defaults below.

    export E5_MODEL_PATH=/path/to/multilingual-e5-base
    export LLAMA_MODEL_PATH=/path/to/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf
"""
import os

# Local sentence-transformers directory for the embedding model. MUST be the same
# model that built data/embeddings.npy (recorded in data/embeddings.meta.json).
E5_MODEL_PATH = os.environ.get(
    "E5_MODEL_PATH",
    "/Users/angelo/Documents/ENS/models/multilingual-e5-base",
)

# Local LLaMA GGUF used to generate the French answers.
LLAMA_MODEL_PATH = os.environ.get(
    "LLAMA_MODEL_PATH",
    "/Users/angelo/Documents/ENS/models/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf",
)

# LLaMA context window. Big enough for the retrieved chunks + prompt + answer.
LLAMA_N_CTX = int(os.environ.get("LLAMA_N_CTX", "8192"))
