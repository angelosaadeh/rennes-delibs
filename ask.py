"""RAG answer step: question -> retrieve chunks -> LLaMA writes a French answer.

    python ask.py "Quelles aides pour le velo ?"   # one-shot
    python ask.py                                   # interactive

The model only ever sees the retrieved extracts, so answers stay grounded in the
actual deliberations instead of the model's parametric guesses. app.py reuses the
helpers below to drive the Gradio chatbot.
"""
import sys

from llama_cpp import Llama

from config import LLAMA_MODEL_PATH, LLAMA_N_CTX
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


def build_llm(n_ctx=LLAMA_N_CTX):
    # n_gpu_layers=-1 offloads everything to the Mac's Metal GPU.
    return Llama(model_path=LLAMA_MODEL_PATH, n_ctx=n_ctx, n_gpu_layers=-1, verbose=False)


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


def stream_answer(llm, messages, max_tokens=MAX_TOKENS, temperature=TEMPERATURE):
    """Yield the answer piece by piece — used for live output in CLI and chatbot."""
    for part in llm.create_chat_completion(
        messages=messages, temperature=temperature, max_tokens=max_tokens, stream=True
    ):
        delta = part["choices"][0]["delta"].get("content")
        if delta:
            yield delta


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
    print("Chargement de l'index et du modèle...", file=sys.stderr)
    retriever = Retriever()
    llm = build_llm()

    def run(q):
        results = retriever.search(q)  # dynamic k: count follows the matches
        messages = build_messages(q, results)
        print()
        for delta in stream_answer(llm, messages):
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
