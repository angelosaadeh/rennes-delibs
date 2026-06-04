# rennes-delibs

Poser des questions en langage naturel sur les délibérations de la **Ville de
Rennes** et de **Rennes Métropole**. Le système télécharge les délibérations
publiques, les indexe, et répond via un modèle de langage local (LLaMA 3.1 8B) en
s'appuyant **uniquement** sur les extraits retrouvés (RAG) — pas d'invention.

Tout tourne **en local** : aucun appel à une API externe, aucune donnée envoyée
ailleurs.

---

## 1. Prérequis

### a) Environnement Python et librairies

```bash
python3 -m venv ~/myenv          # un environnement virtuel (si vous n'en avez pas)
~/myenv/bin/pip install -r requirements.txt
```

### b) Les deux modèles

```bash
~/myenv/bin/python3 download_models.py
```

Télécharge l'embedding `multilingual-e5-base` (~1 Go) et le modèle de génération
`Meta-Llama-3.1-8B-Instruct-Q6_K.gguf` (~6,6 Go). Les fichiers déjà présents sont
ignorés.

Les chemins des modèles sont définis dans **`config.py`** et surchargés par
variables d'environnement — utile pour les placer ailleurs :

```bash
export E5_MODEL_PATH=/chemin/vers/multilingual-e5-base
export LLAMA_MODEL_PATH=/chemin/vers/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf
```

---

## 2. Construire (ou mettre à jour) l'index

```bash
~/myenv/bin/python3 update.py
```

Une seule commande enchaîne tout le pipeline, de façon **incrémentale** (seules
les nouvelles délibérations sont traitées) :

1. `fetch_catalog.py` — récupère le catalogue à jour depuis l'open data de Rennes
2. `download_pdfs.py` — télécharge les PDFs manquants
3. `extract_text.py` — extrait le texte et la liste des présents
4. `chunk.py` — découpe en chunks pour le RAG
5. `embed.py` — calcule les vecteurs d'embedding (`data/embeddings.npy`)

Relancez `update.py` quand de nouvelles délibérations sont publiées : seuls les
nouveaux chunks sont embeddés (quelques secondes au lieu de tout recalculer).

---

## 3. Poser des questions

### Chatbot (recommandé)

```bash
~/myenv/bin/python3 app.py
```

Ouvrez ensuite **http://localhost:7860**. La réponse s'affiche au fil de l'eau,
suivie des délibérations sources.

### En ligne de commande

```bash
~/myenv/bin/python3 ask.py "Quelles décisions sur le vélo et les pistes cyclables ?"
~/myenv/bin/python3 ask.py          # mode interactif
```

---

## Comment ça marche

- **Récupération** (`retrieve.py`) : la question est embeddée avec **le même**
  modèle e5 que les délibérations, puis comparée par similarité cosinus. Le
  nombre d'extraits retenus est **dynamique** — il s'adapte au nombre de
  correspondances pertinentes (coupe au plus grand décrochage de score).
- **Génération** (`ask.py`) : les extraits retenus sont fournis à LLaMA, à qui
  l'on demande de répondre uniquement à partir d'eux, en français, avec sources.

Tous les fichiers dérivés (PDFs, JSON extraits, index) sont dans `data/` et
ignorés par git.
