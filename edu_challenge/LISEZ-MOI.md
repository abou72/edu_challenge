# EDU-PREDICT BURKINA - Data Challenge Burkina 2026, Défi 1

Cartographie et anticipation du risque de décrochage scolaire par région (primaire, 13 régions, 2015/16-2024/25).

## Contenu
- `app.py` : web app Streamlit (carte du risque, fiche région, prévision et alertes, méthode et limites)
- `analyse.py` : indice composite de risque, test de robustesse, comparaison de modèles, prévision 2025/26
- `pipeline/` : 01_nettoyage.py, 02_fusion.py, 03_transformation.py
- `data/` : fichiers produits (tables propres, table maître, variables, indice, prévisions, dictionnaire des variables)
- `requirements.txt` : bibliothèques Python

## Relancer tout le travail (Python 3.11)
```
pip install -r requirements.txt
# 1. placer le fichier Excel DGESS dans data/raw/
python pipeline/01_nettoyage.py      # 7 CSV propres + journal_nettoyage.txt
python pipeline/02_fusion.py         # attendu : (130, 23) 0 valeurs vides
python pipeline/03_transformation.py # attendu : (130, 33)
python analyse.py                    # indice, robustesse, validation, prévisions
streamlit run app.py
```
Carte géographique (optionnel) : placer le GeoJSON des 13 régions dans `data/regions_bf.geojson` (Data Pack, OCHA). Sans lui, l'app affiche un cartogramme.

## Source
DGESS, Ministère de l'Éducation nationale : Données statistiques préscolaire et primaire 2016-2025. Données utilisées uniquement pour le concours.

## Usage de l'IA
Un assistant IA (Claude) a aidé à écrire les scripts, l'application et le brouillon de la documentation à partir de la structure du fichier. Les données publiques agrégées ont été lues par l'assistant ; l'auteur a exécuté le nettoyage et vérifié les résultats. À compléter par l'auteur : ce qu'il a demandé, ce qu'il a retenu ou corrigé.
