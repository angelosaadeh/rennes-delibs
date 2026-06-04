"""Gradio chatbot over the Rennes deliberations.

    python app.py        # then open http://localhost:7860

Loads the embedding index and LLaMA once at startup, then answers each message by
retrieving the relevant chunks (dynamic count) and streaming a grounded French
answer, followed by the source deliberations.
"""
import os
import signal

import gradio as gr

from ask import build_messages, format_sources, load_backend, stream_answer
from config import LLM_BACKEND, LOG_DATASET
from retrieve import Retriever

# Exit immediately on Ctrl+C. The native ML libs (torch/Metal, llama.cpp) tend to
# segfault during interpreter teardown, which pops a macOS "Python quit
# unexpectedly" dialog. os._exit skips that cleanup so shutdown is clean.
signal.signal(signal.SIGINT, lambda *_: os._exit(0))

print(f"Chargement de l'index et du modèle (backend: {LLM_BACKEND})...")
retriever = Retriever()
backend = load_backend()

# Optional analytics: append each interaction to a JSONL that a CommitScheduler
# pushes to a PRIVATE HF Dataset every few minutes. Off unless LOG_DATASET is set
# (with an HF_TOKEN secret that can write to it). No IP is recorded.
_log = None
if LOG_DATASET:
    import json
    import uuid
    from datetime import datetime, timezone
    from pathlib import Path

    from huggingface_hub import CommitScheduler

    _log_dir = Path("logs")
    _log_dir.mkdir(exist_ok=True)
    _log_file = _log_dir / f"chat-{uuid.uuid4()}.jsonl"
    _scheduler = CommitScheduler(
        repo_id=LOG_DATASET, repo_type="dataset", folder_path=_log_dir,
        every=10, private=True,
    )

    def _log(question, results, answer):
        with _scheduler.lock, _log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": datetime.now(timezone.utc).isoformat(),
                "backend": LLM_BACKEND,
                "question": question,
                "sources": [c["delib_id"] for _, c in results],
                "scores": [round(s, 3) for s, _ in results],
                "answer": answer,
            }, ensure_ascii=False) + "\n")

    print(f"Journalisation activée → dataset privé {LOG_DATASET}")


def respond(message, history):
    results = retriever.search(message)  # dynamic k
    messages = build_messages(message, results)
    answer = ""
    for delta in stream_answer(backend, messages):
        answer += delta
        yield answer
    if _log:
        try:
            _log(message, results, answer)
        except Exception as e:  # never let logging break a reply
            print(f"log error: {e}")
    yield answer + "\n\n---\n**Sources :**\n" + format_sources(results)


DESCRIPTION = (
    "Posez une question ; les réponses s'appuient uniquement sur les "
    "délibérations indexées (2021→). Les sources sont listées sous chaque réponse."
)
if LOG_DATASET:
    DESCRIPTION += (
        " ⓘ Les questions et réponses peuvent être enregistrées de manière "
        "confidentielle afin d'améliorer l'outil."
    )

demo = gr.ChatInterface(
    respond,
    title="Délibérations de Rennes & Rennes Métropole",
    description=DESCRIPTION,
    examples=[
        "Quelles décisions concernant le vélo et les pistes cyclables ?",
        "Quelles subventions pour la culture et les festivals ?",
        "Qu'a décidé le conseil sur le Plan Climat Air Énergie ?",
    ],
    # Don't pre-run the examples at startup — that would call Groq before any user
    # interaction and make the whole app fail to launch if a call errors.
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch()
