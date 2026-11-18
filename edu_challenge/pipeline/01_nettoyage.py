"""Nettoyage des tableaux du primaire (DGESS 2015/16 - 2024/25), Défi 1.

Usage :  python nettoyage_primaire.py chemin/vers/fichier.xlsx [dossier_sortie]
Sortie : un CSV "long" par tableau + journal_nettoyage.txt

Format long : region ; annee_debut ; (sexe | statut) ; valeur ; est_national
annee_debut = année de rentrée (2015 pour 2015/2016).
"""
import re
import sys
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent.parent / "data" / "raw" / "DONNEES_STATISTQUES__PRESCOLAIRE_ET_PRIMAIRE_2016_2025_DCB.xlsx"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent.parent / "data"
OUT.mkdir(exist_ok=True)
LOG = []

NATIONAL = {"NATIONAL", "BURKINA FASO"}
SEXE = {"GARCONS": "G", "FILLES": "F", "TOTAL": "T"}


def log(msg):
    LOG.append(msg)
    print(msg)


def norm_region(x):
    r = re.sub(r"\s+", " ", str(x)).strip().upper()
    return {"HAUT-BASSINS": "HAUTS-BASSINS", "NATIONAL": "BURKINA FASO"}.get(r, r)


def annee_debut(label, civile=False):
    """Extrait l'année du libellé ('2018-2019', '2015/16', 2016...) sans le texte parasite."""
    m = re.search(r"(20\d\d)", str(label))
    if not m:
        return None
    y = int(m.group(1))
    return y - 1 if civile else y  # CEP / préscolaire : année civile de fin


def lire(ws):
    return [list(r) for r in ws.iter_rows(values_only=True)]


def long_simple(rows, cle2, civile=False):
    """Tableaux 'REGION | SEXE ou STATUT | années...' (région renseignée une ligne sur trois)."""
    header = rows[1]
    cols = {j: annee_debut(h, civile) for j, h in enumerate(header) if j >= 2 and annee_debut(h, civile)}
    out, region = [], None
    for r in rows[2:]:
        if r[0] is not None:
            region = norm_region(r[0])
        if region is None or r[1] is None or not re.match(r"^[A-Za-zÉé]", str(r[1])):
            continue
        if region not in NATIONAL and not re.match(r"^[A-Z\- ]+$", region):
            continue
        for j, a in cols.items():
            v = pd.to_numeric(r[j], errors="coerce")
            out.append({"region": region, "annee_debut": a, cle2: str(r[1]).strip().upper(),
                        "valeur": v, "est_national": region in NATIONAL})
    return pd.DataFrame(out)


def long_large(rows):
    """Tableaux TBA / TBS / Achèvement : années sur 2 lignes, colonnes G/F/T par année."""
    ann, cur = {}, None
    for j, h in enumerate(rows[1]):
        if j >= 1 and annee_debut(h):
            cur = annee_debut(h)
        if cur and j >= 1:
            ann[j] = (cur, rows[2][j])
    out = []
    for r in rows[3:]:
        if r[0] is None:
            if out:
                break  # fin du premier bloc
            continue
        region = norm_region(r[0])
        if region not in NATIONAL and not re.match(r"^[A-Z\- ]+$", region):
            break
        for j, (a, s) in ann.items():
            out.append({"region": region, "annee_debut": a, "sexe": s,
                        "valeur": pd.to_numeric(r[j], errors="coerce"),
                        "est_national": region in NATIONAL})
        if region in NATIONAL:
            break
    return pd.DataFrame(out)


wb = load_workbook(SRC, read_only=True, data_only=True)

TABLES = {  # feuille : (nom de sortie, fonction)
    "Feuil1":  ("effectifs_region_sexe",   lambda r: long_simple(r, "sexe")),
    "Feuil5":  ("nouveaux_cp1_region_sexe", lambda r: long_simple(r, "sexe")),
    "Feuil6":  ("tba_region_sexe",          long_large),
    "Feuil7":  ("tbs_region_sexe",          long_large),
    "Feuil8":  ("achevement_region_sexe",   long_large),
    "Feuil9":  ("cep_region_sexe",          lambda r: long_simple(r, "sexe", civile=True)),
    "Feuil13": ("ecoles_region_statut",     lambda r: long_simple(r, "statut")),
}

for feuille, (nom, fn) in TABLES.items():
    df = fn(lire(wb[feuille]))
    if "sexe" in df:
        df["sexe"] = df["sexe"].replace({"GARCONS": "G", "GARÇONS": "G", "FILLES": "F", "TOTAL": "T"})
    n_nan = int(df["valeur"].isna().sum())
    log(f"{feuille} -> {nom}.csv : {len(df)} lignes, {df['region'].nunique()} zones, "
        f"années {df['annee_debut'].min()}-{df['annee_debut'].max()}, {n_nan} valeurs manquantes/non numériques")
    df.to_csv(OUT / f"{nom}.csv", index=False, encoding="utf-8")

log("Choix : annee_debut = année de rentrée ; CEP (année civile) décalé d'un an ; "
    "HAUT-BASSINS renommé HAUTS-BASSINS ; indicateurs IPS (Feuil10-12) non repris : à recalculer (F/G).")
log("Texte parasite adressé à une IA repéré dans des en-têtes/cellules : ignoré (années extraites par expression régulière).")
(OUT / "journal_nettoyage.txt").write_text("\n".join(LOG), encoding="utf-8")
