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
#   "groq"  — Groq's hosted GPT-OSS 120B (no big download; needs a free key
#             from https://console.groq.com and `pip install groq`)
# .strip() throughout: HF Space secrets/variables often keep a trailing newline,
# and a newline in the API key produces an "illegal header value" on every call.
LLM_BACKEND = os.environ.get("LLM_BACKEND", "local").strip().lower()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
# Groq retired its Llama models for free/developer keys on 2026-08-16
# (llama-3.3-70b-versatile -> openai/gpt-oss-120b, llama-3.1-8b-instant ->
# openai/gpt-oss-20b). GPT-OSS is a reasoning model, so GROQ_REASONING_EFFORT
# (low|medium|high) trades answer quality against tokens/latency.
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b").strip()
GROQ_REASONING_EFFORT = os.environ.get("GROQ_REASONING_EFFORT", "low").strip().lower()
# Completion budget (hidden reasoning + visible answer). Counts toward the TPM
# limit, so keep it modest on the free tier; raise it on a paid key.
GROQ_MAX_TOKENS = int(os.environ.get("GROQ_MAX_TOKENS", "2048"))

# --- Analytics (optional) --------------------------------------------------
# If set to a dataset repo id (e.g. "angelosaadeh/rennes-delibs-logs"), each
# interaction is appended to a PRIVATE HF Dataset for analytics. Needs an
# HF_TOKEN secret with write access. Empty = no logging (the default).
LOG_DATASET = os.environ.get("LOG_DATASET", "").strip()

# --- Retrieval breadth -----------------------------------------------------
# Number of chunks returned per question (top-k, incl. sibling expansion).
# The binding limit on Groq is NOT the 128k context but the FREE tier's 8k
# tokens/minute for gpt-oss-120b: prompt + max_tokens of a single request must
# fit under it. At ~500 tokens/chunk, 8 chunks (~4.5k) + a 2k answer budget fits.
# On a paid (developer) key — 250k TPM — raise this, e.g. RETRIEVAL_MAX_K=100.
RETRIEVAL_MAX_K = int(
    os.environ.get("RETRIEVAL_MAX_K", "8")
)
