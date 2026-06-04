"""Settings in one place.

Values come from (in order of precedence): real environment variables, then a
local `.env` file (copy `.env.example` to `.env` and edit), then the defaults
below. So you configure things once in `.env` and just run `python app.py` —
no need to pass env vars on the command line.
"""
import os


def _load_dotenv(path=os.path.join(os.path.dirname(__file__), ".env")):
    """Minimal .env reader (KEY=VALUE lines), no external dependency.

    Uses setdefault so a real exported environment variable always wins over the
    file — handy for one-off overrides and for hosting (e.g. HF Space secrets).
    """
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_dotenv()

# --- Model locations -------------------------------------------------------
# Defaults are project-local; point them elsewhere via .env to reuse existing
# files. The e5 model MUST match the one that built data/embeddings.npy.
E5_MODEL_PATH = os.environ.get("E5_MODEL_PATH", "models/multilingual-e5-base")
LLAMA_MODEL_PATH = os.environ.get(
    "LLAMA_MODEL_PATH", "models/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf"
)
LLAMA_N_CTX = int(os.environ.get("LLAMA_N_CTX", "8192"))

# --- Generation backend ----------------------------------------------------
#   "local" — the local GGUF above (fully offline, default)
#   "groq"  — Groq's free hosted Llama 3.1 8B (no big download; needs a free key
#             from https://console.groq.com and `pip install groq`)
LLM_BACKEND = os.environ.get("LLM_BACKEND", "local").lower()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
# Llama 3.3 70B is free on Groq and far stronger than the local 8B — worth using
# since generation is hosted. Override with GROQ_MODEL (e.g. llama-3.1-8b-instant).
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
