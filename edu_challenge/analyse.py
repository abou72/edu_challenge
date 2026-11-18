"""EDU-PREDICT BURKINA - Défi 1 : indice de risque de décrochage par région + prévision.

Entrée  : data/table_variables.csv (13 régions x 10 ans, DGESS primaire 2015/16-2024/25)
Sorties : data/indice_risque.csv, data/sensibilite.csv, data/validation_modeles.csv,
          data/previsions_2025.csv, data/resume_chiffres.json
Lancer  : python analyse.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

D = Path(__file__).parent / "data"
m = pd.read_csv(D / "table_variables.csv").sort_values(["region", "annee_debut"]).reset_index(drop=True)
g = m.groupby("region")

# ------------------------------------------------------------------ 1. Indicateurs de l'indice
m["ecoles_perdues_pic"] = (g.ecoles_total.cummax() - m.ecoles_total) / g.ecoles_total.cummax() * 100
m["tendance_achev_3a"] = (m.achev_T - g.achev_T.shift(3)) / 3  # points par an sur 3 ans

POIDS = {"achev_T": 0.35, "ecart_tba_achev": 0.20, "ecoles_perdues_pic": 0.20,
         "tendance_achev_3a": 0.15, "eleves_par_ecole": 0.10}
SENS = {"achev_T": -1, "ecart_tba_achev": +1, "ecoles_perdues_pic": +1,
        "tendance_achev_3a": -1, "eleves_par_ecole": +1}  # +1 : valeur haute = risque haut


def indice(df, poids):
    """Min-max entre régions (0 = meilleur, 1 = pire), puis moyenne pondérée, sur 0-100."""
    s = pd.Series(0.0, index=df.index)
    for c, w in poids.items():
        lo, hi = df[c].min(), df[c].max()
        x = (df[c] - lo) / (hi - lo) if hi > lo else df[c] * 0
        s += w * (x if SENS[c] > 0 else 1 - x)
    return s * 100


def niveau(v):
    return "Critique" if v >= 65 else "Élevé" if v >= 50 else "Modéré" if v >= 35 else "Faible"


parts = []
for y, df in m[m.annee_debut >= 2018].groupby("annee_debut"):
    df = df.copy()
    df["indice_risque"] = indice(df, POIDS)
    df["rang"] = df.indice_risque.rank(ascending=False).astype(int)
    df["niveau"] = df.indice_risque.map(niveau)
    parts.append(df)
idx = pd.concat(parts)
cols = ["region", "annee_debut", "indice_risque", "rang", "niveau"] + list(POIDS) + \
       ["achev_T", "achev_F", "achev_G", "tba_T", "cep_T", "ecoles_total", "effectifs_T"]
idx[cols].round(2).to_csv(D / "indice_risque.csv", index=False)

# ------------------------------------------------------------------ 2. Sensibilité (année 2024/25)
d24 = m[m.annee_debut == 2024].copy()
base = indice(d24, POIDS)
rng = np.random.default_rng(42)
ranks, rho = [], []
for _ in range(2000):
    w = rng.dirichlet(np.ones(len(POIDS)))
    s = indice(d24, dict(zip(POIDS, w)))
    ranks.append(s.rank(ascending=False).values)
    rho.append(spearmanr(base, s)[0])
ranks = np.array(ranks)
eq = indice(d24, {c: 1 / len(POIDS) for c in POIDS})
sens = pd.DataFrame({
    "region": d24.region.values,
    "rang_base": base.rank(ascending=False).astype(int).values,
    "rang_poids_egaux": eq.rank(ascending=False).astype(int).values,
    "rang_moyen_2000_tirages": ranks.mean(0).round(2),
    "part_top3_pct": ((ranks <= 3).mean(0) * 100).round(1),
    "part_top5_pct": ((ranks <= 5).mean(0) * 100).round(1),
}).sort_values("rang_base")
top5_base = set(base.nlargest(5).index)
top5_identique = float(np.mean([set(d24.index[np.argsort(r)[:5]]) == top5_base for r in ranks]) * 100)
sens.to_csv(D / "sensibilite.csv", index=False)

# ------------------------------------------------------------------ 3. Prévision de l'achèvement
m["cible"] = g.achev_T.shift(-1)          # achèvement de l'année suivante
m["delta"] = m.cible - m.achev_T
FEATS = ["achev_T", "var_achev", "var_ecoles_pct", "ecart_tba_achev", "ecoles_perdues_pic"]
m["drift3"] = (m.achev_T - g.achev_T.shift(3)) / 3


def predire(train, test, nom):
    if nom == "naif":
        return test.achev_T.values
    if nom == "tendance_3a":
        return (test.achev_T + test.drift3).values
    sc = StandardScaler().fit(train[FEATS])
    r = Ridge(alpha=10.0).fit(sc.transform(train[FEATS]), train.delta)
    return test.achev_T.values + r.predict(sc.transform(test[FEATS]))


MODELES = ["naif", "tendance_3a", "ridge_panel"]
lignes, erreurs = [], {k: [] for k in MODELES}
for Y in (2022, 2023, 2024):                       # année cible testée
    train = m[(m.annee_debut + 1 <= Y - 1)].dropna(subset=FEATS + ["delta"])
    test = m[m.annee_debut == Y - 1].dropna(subset=["cible", "drift3"])
    for k in MODELES:
        e = test.cible.values - predire(train, test, k)
        erreurs[k] += list(e)
        lignes.append({"modele": k, "annee_cible": Y, "MAE": np.abs(e).mean(), "n": len(e)})
val = pd.DataFrame(lignes)
tot = pd.DataFrame({"modele": MODELES,
                    "annee_cible": "2022-2024",
                    "MAE": [np.abs(erreurs[k]).mean() for k in MODELES],
                    "RMSE": [np.sqrt(np.mean(np.square(erreurs[k]))) for k in MODELES],
                    "n": [len(erreurs[k]) for k in MODELES]})
pd.concat([val, tot]).round(3).to_csv(D / "validation_modeles.csv", index=False)
meilleur = tot.sort_values("MAE").modele.iloc[0]

# prévision 2025/26 avec le meilleur modèle, intervalle empirique 80 %
last = m[m.annee_debut == 2024].copy()
train_all = m.dropna(subset=FEATS + ["delta"])
last["prevision_achev_2025"] = predire(train_all, last, meilleur)
e = np.array(erreurs[meilleur])
q10, q90 = np.quantile(e, [0.10, 0.90])
last["borne_basse_80"] = last.prevision_achev_2025 + q10
last["borne_haute_80"] = last.prevision_achev_2025 + q90
last["variation_prevue"] = last.prevision_achev_2025 - last.achev_T
prev = last.merge(idx[idx.annee_debut == 2024][["region", "indice_risque", "niveau"]], on="region")
haut = prev.niveau.isin(["Critique", "Élevé"])
degrade = (prev.tendance_achev_3a < 0) | (prev.var_ecoles_pct < -10)
prev["alerte"] = np.select(
    [haut & degrade, haut, (prev.niveau == "Modéré") & degrade],
    ["Priorité 1 : risque élevé et en dégradation", "Priorité 2 : risque élevé, stable",
     "Veille renforcée"], default="Veille")
prev[["region", "achev_T", "prevision_achev_2025", "borne_basse_80", "borne_haute_80", "variation_prevue",
      "indice_risque", "niveau", "tendance_achev_3a", "var_ecoles_pct", "alerte"]].round(2).sort_values("indice_risque", ascending=False) \
    .to_csv(D / "previsions_2025.csv", index=False)

# ------------------------------------------------------------------ 4. Chiffres clés
i24 = idx[idx.annee_debut == 2024].sort_values("rang")
nat = pd.read_csv(D / "table_nationale_controle.csv") if (D / "table_nationale_controle.csv").exists() else None
resume = {
    "meilleur_modele": meilleur,
    "mae_par_modele": tot.set_index("modele").MAE.round(2).to_dict(),
    "intervalle_80_points": [round(q10, 2), round(q90, 2)],
    "top5_2024": i24.head(5)[["region", "indice_risque", "niveau"]].round(1).to_dict("records"),
    "rho_spearman_poids_aleatoires_moyen": round(float(np.mean(rho)), 3),
    "rho_spearman_poids_aleatoires_p5": round(float(np.quantile(rho, 0.05)), 3),
    "top5_identique_pct_des_tirages": round(top5_identique, 1),
    "achev_moyen_regions_2015": round(float(m[m.annee_debut == 2015].achev_T.mean()), 1),
    "achev_moyen_regions_2024": round(float(m[m.annee_debut == 2024].achev_T.mean()), 1),
}
(D / "resume_chiffres.json").write_text(json.dumps(resume, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(resume, ensure_ascii=False, indent=2))
