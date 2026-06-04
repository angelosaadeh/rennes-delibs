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
from config import LLM_BACKEND
from retrieve import Retriever

# Exit immediately on Ctrl+C. The native ML libs (torch/Metal, llama.cpp) tend to
# segfault during interpreter teardown, which pops a macOS "Python quit
# unexpectedly" dialog. os._exit skips that cleanup so shutdown is clean.
signal.signal(signal.SIGINT, lambda *_: os._exit(0))

print(f"Chargement de l'index et du modèle (backend: {LLM_BACKEND})...")
retriever = Retriever()
backend = load_backend()


def respond(message, history):
    results = retriever.search(message)  # dynamic k
    messages = build_messages(message, results)
    answer = ""
    for delta in stream_answer(backend, messages):
        answer += delta
        yield answer
    yield answer + "\n\n---\n**Sources :**\n" + format_sources(results)


demo = gr.ChatInterface(
    respond,
    title="Délibérations de Rennes & Rennes Métropole",
    description=(
        "Posez une question ; les réponses s'appuient uniquement sur les "
        "délibérations indexées (2021→). Les sources sont listées sous chaque réponse."
    ),
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
