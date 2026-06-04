"""RAG answer step: question -> retrieve chunks -> an LLM writes a French answer.

    python ask.py "Quelles aides pour le velo ?"   # one-shot
    python ask.py                                   # interactive

Generation goes through a pluggable backend (config.LLM_BACKEND):
  - "local": the local LLaMA GGUF, fully offline (default)
  - "groq" : Groq's free hosted Llama 3.1 8B (no 6.6 GB download)
Either way the model only ever sees the retrieved extracts, so answers stay
grounded in the actual deliberations. app.py reuses the helpers below.
"""
import sys

from config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    LLAMA_MODEL_PATH,
    LLAMA_N_CTX,
    LLM_BACKEND,
)
from retrieve import Retriever

MAX_TOKENS = 1024
TEMPERATURE = 0.2

SYSTEM = (
    "Tu es un assistant qui répond à des questions sur les délibérations des "
    "conseils de la Ville de Rennes et de Rennes Métropole. Réponds UNIQUEMENT à "
    "partir des extraits fournis, en français. Si la réponse ne figure pas dans "
    "les extraits, dis-le clairement plutôt que d'inventer. Quand tu cites une "
    "décision, mentionne son objet et sa date."
)


class LocalBackend:
    """Local LLaMA GGUF via llama-cpp-python (offline)."""

    def __init__(self):
        from llama_cpp import Llama  # imported lazily: only needed for this backend

        # n_gpu_layers=-1 offloads everything to the Mac's Metal GPU.
        self.llm = Llama(
            model_path=LLAMA_MODEL_PATH, n_ctx=LLAMA_N_CTX, n_gpu_layers=-1, verbose=False
        )

    def stream(self, messages, max_tokens, temperature):
        for part in self.llm.create_chat_completion(
            messages=messages, temperature=temperature, max_tokens=max_tokens, stream=True
        ):
            delta = part["choices"][0]["delta"].get("content")
            if delta:
                yield delta


class GroqBackend:
    """Groq's hosted Llama 3.1 8B (free tier, OpenAI-compatible)."""

    def __init__(self):
        from groq import Groq  # imported lazily: only needed for this backend

        if not GROQ_API_KEY:
            raise SystemExit(
                "LLM_BACKEND=groq but GROQ_API_KEY is not set. "
                "Get a free key at https://console.groq.com and export GROQ_API_KEY."
            )
        self.client = Groq(api_key=GROQ_API_KEY)

    def stream(self, messages, max_tokens, temperature):
        stream = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


def load_backend():
    if LLM_BACKEND == "local":
        return LocalBackend()
    if LLM_BACKEND == "groq":
        return GroqBackend()
    raise SystemExit(f"Unknown LLM_BACKEND={LLM_BACKEND!r} (use 'local' or 'groq').")


def build_messages(question, results):
    blocks = [
        f"[Extrait {i} — {c['delib_objet']} ({c['delib_date']})]\n{c['text']}"
        for i, (_score, c) in enumerate(results, 1)
    ]
    context = "\n\n".join(blocks)
    user = f"Extraits des délibérations :\n\n{context}\n\nQuestion : {question}"
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]


def stream_answer(backend, messages, max_tokens=MAX_TOKENS, temperature=TEMPERATURE):
    """Yield the answer piece by piece — used for live output in CLI and chatbot."""
    return backend.stream(messages, max_tokens, temperature)


def format_sources(results):
    """Markdown bullet list, one line per deliberation (deduped across chunks)."""
    seen, lines = set(), []
    for score, c in results:
        if c["delib_id"] in seen:
            continue
        seen.add(c["delib_id"])
        lines.append(f"- {c['delib_objet']} ({c['delib_date']})  _[score {score:.2f}]_")
    return "\n".join(lines)


def main():
    question = " ".join(sys.argv[1:]).strip()
    print(f"Chargement (backend: {LLM_BACKEND})...", file=sys.stderr)
    retriever = Retriever()
    backend = load_backend()

    def run(q):
        results = retriever.search(q)  # dynamic k: count follows the matches
        messages = build_messages(q, results)
        print()
        for delta in stream_answer(backend, messages):
            sys.stdout.write(delta)
            sys.stdout.flush()
        print("\n\nSources :")
        print(format_sources(results))

    if question:
        run(question)
        return

    print("Pose une question sur les délibérations (Ctrl-D pour quitter).")
    while True:
        try:
            q = input("\n> ").strip()
        except EOFError:
            print()
            break
        if q:
            run(q)


if __name__ == "__main__":
    main()
