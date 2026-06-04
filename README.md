# rennes-delibs

Poser des questions en langage naturel sur les délibérations de la **Ville de
Rennes** et de **Rennes Métropole**. Le système retrouve les extraits pertinents
(RAG) et un modèle Llama 3 rédige la réponse en français, en s'appuyant
**uniquement** sur ces extraits — pas d'invention.

L'index des délibérations est **fourni dans le dépôt** : pour simplement poser des
questions, inutile de re-télécharger ou ré-indexer quoi que ce soit.

---

## Installation

```bash
python3 -m venv ~/myenv
~/myenv/bin/pip install -r requirements.txt
cp .env.example .env          # votre configuration (chemins, backend, clé)
```

Vous configurez tout dans `.env`, puis vous lancez simplement `python app.py`.

Choisissez **où** le modèle génère les réponses :

### Option A — Groq (léger, et meilleur modèle)

Réponses générées par **Llama 3.3 70B** hébergé gratuitement par Groq — bien plus
puissant que le 8B local, et **aucun téléchargement de 6,6 Go**.

```bash
~/myenv/bin/pip install groq
```

Dans `.env` :

```ini
LLM_BACKEND=groq
GROQ_API_KEY=gsk_...          # clé gratuite : https://console.groq.com
```

Puis :

```bash
~/myenv/bin/python3 download_models.py   # ne télécharge que l'embedding e5 (~1 Go)
~/myenv/bin/python3 app.py
```

### Option B — Tout en local (hors-ligne, privé)

Réponses générées par un Llama 3.1 8B local. Rien ne sort de votre machine.

```bash
~/myenv/bin/pip install llama-cpp-python
```

Dans `.env`, laissez `LLM_BACKEND=local`, puis :

```bash
~/myenv/bin/python3 download_models.py   # e5 (~1 Go) + Llama GGUF (~6,6 Go)
~/myenv/bin/python3 app.py
```

Dans les deux cas, ouvrez **http://localhost:7860**. La réponse s'affiche au fil
de l'eau, suivie des délibérations sources.

> L'embedding e5 est requis dans tous les cas : il transforme **votre question**
> en vecteur pour la recherche. Par défaut les modèles sont cherchés dans
> `./models/` ; pointez `E5_MODEL_PATH` / `LLAMA_MODEL_PATH` (dans `.env`) vers
> des fichiers existants pour éviter de retélécharger.

### En ligne de commande

```bash
~/myenv/bin/python3 ask.py "Quelles décisions sur le vélo et les pistes cyclables ?"
~/myenv/bin/python3 ask.py          # mode interactif
```

---

## Mettre à jour l'index (optionnel)

Pour intégrer de nouvelles délibérations publiées depuis la dernière indexation :

```bash
~/myenv/bin/python3 update.py
```

Une seule commande enchaîne tout le pipeline, de façon **incrémentale** (seules
les nouvelles délibérations sont traitées) :

1. `fetch_catalog.py` — catalogue à jour depuis l'open data de Rennes
2. `download_pdfs.py` — télécharge les PDFs manquants
3. `extract_text.py` — extrait le texte et la liste des présents
4. `chunk.py` — découpe en chunks (`data/chunks.json.gz`)
5. `embed.py` — calcule les embeddings (`data/embeddings.npy`)

Seuls les nouveaux chunks sont embeddés (quelques secondes), pas tout le corpus.

---

## Comment ça marche

- **Récupération** (`retrieve.py`) : la question est embeddée avec **le même**
  modèle e5 que les délibérations, puis comparée par similarité cosinus. Le
  nombre d'extraits retenus est **dynamique** — il s'adapte au nombre de
  correspondances pertinentes (coupe au plus grand décrochage de score).
- **Génération** (`ask.py`) : les extraits sont fournis au LLM (local ou Groq),
  à qui l'on demande de répondre uniquement à partir d'eux, en français, avec
  les sources.

L'index est versionné (`data/chunks.json.gz`, `data/embeddings.npy`) ; les
fichiers intermédiaires (PDFs, JSON extraits) restent dans `data/` et sont
ignorés par git.
