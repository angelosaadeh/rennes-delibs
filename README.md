# Rennes LEO

Système permettant d'interroger les délibérations de la Ville de Rennes et de Rennes Métropole.

---

## Étape 1 — Téléchargement des PDFs

```bash
python3 download_pdfs.py
```

Télécharge les PDFs dans `data/pdfs-ville/` et `data/pdfs-metropole/`. Certains fichiers peuvent échouer — erreur côté serveur Mégalis Bretagne.

## Étape 2 — Extraction du texte

```bash
python3 extract_text.py
```

Extrait le texte de chaque PDF et la liste des présents/procurations/absents. Produit `data/extracted-ville.json` et `data/extracted-metropole.json`.

## Étape 3 — Découpage en chunks

```bash
python3 chunk.py
```

Découpe chaque délibération en chunks pour le RAG. La liste des présents forme un chunk dédié, le corps de la décision est découpé séparément. Produit `data/chunks.json`.
