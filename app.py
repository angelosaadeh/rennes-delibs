"""Gradio chatbot over the Rennes deliberations.

    python app.py        # then open http://localhost:7860

Loads the embedding index and LLaMA once at startup, then answers each message by
retrieving the relevant chunks (dynamic count) and streaming a grounded French
answer, followed by the source deliberations.
"""
import gradio as gr

from ask import build_llm, build_messages, format_sources, stream_answer
from retrieve import Retriever

print("Chargement de l'index et du modèle LLaMA (~10-20 s)...")
retriever = Retriever()
llm = build_llm()


def respond(message, history):
    results = retriever.search(message)  # dynamic k
    messages = build_messages(message, results)
    answer = ""
    for delta in stream_answer(llm, messages):
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
)

if __name__ == "__main__":
    demo.launch()
