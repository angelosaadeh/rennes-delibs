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
E5_MODEL_PATH = os.environ.get("E5_MODEL_PATH", "models/multilingual-e5-base").strip()
LLAMA_MODEL_PATH = os.environ.get(
    "LLAMA_MODEL_PATH", "models/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf"
).strip()
LLAMA_N_CTX = int(os.environ.get("LLAMA_N_CTX", "8192"))

# --- Generation backend ----------------------------------------------------
#   "local" — the local GGUF above (fully offline, default)
#   "groq"  — Groq's free hosted Llama 3.1 8B (no big download; needs a free key
#             from https://console.groq.com and `pip install groq`)
# .strip() throughout: HF Space secrets/variables often keep a trailing newline,
# and a newline in the API key produces an "illegal header value" on every call.
LLM_BACKEND = os.environ.get("LLM_BACKEND", "local").strip().lower()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
# Llama 3.3 70B is free on Groq and far stronger than the local 8B — worth using
# since generation is hosted. Override with GROQ_MODEL (e.g. llama-3.1-8b-instant).
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile").strip()

# --- Analytics (optional) --------------------------------------------------
# If set to a dataset repo id (e.g. "angelosaadeh/rennes-delibs-logs"), each
# interaction is appended to a PRIVATE HF Dataset for analytics. Needs an
# HF_TOKEN secret with write access. Empty = no logging (the default).
LOG_DATASET = os.environ.get("LOG_DATASET", "").strip()

# --- Retrieval breadth -----------------------------------------------------
# Dynamic-k ceiling. Groq's hosted 70B has a 128k context, so it can use many
# more chunks than the local 8B (capped by LLAMA_N_CTX) — broader, more complete
# answers. Override with RETRIEVAL_MAX_K.
RETRIEVAL_MIN_K = int(os.environ.get("RETRIEVAL_MIN_K", "3"))
RETRIEVAL_MAX_K = int(
    os.environ.get("RETRIEVAL_MAX_K", "20" if LLM_BACKEND == "groq" else "8")
)
