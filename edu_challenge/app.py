"""EDU-PREDICT BURKINA - tableau de bord d'aide à la décision (Défi 1, Data Challenge Burkina 2026).

Lancer :  streamlit run app.py
Données : dossier data/ (produit par nettoyage_primaire.py, fusion, transformation et analyse.py)
Source  : DGESS, Ministère de l'Éducation nationale - statistiques préscolaire et primaire 2016-2025.
"""
import json
import unicodedata
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

D = Path(__file__).parent / "data"
st.set_page_config(page_title="EDU-PREDICT BURKINA", page_icon="🎓", layout="wide")

# --------------------------------------------------------------------------- paramètres de l'indice
LIBELLES = {
    "achev_T": "Taux d'achèvement (bas = risque)",
    "ecart_tba_achev": "Écart admission - achèvement",
    "ecoles_perdues_pic": "Écoles perdues depuis le pic",
    "tendance_achev_3a": "Tendance de l'achèvement sur 3 ans",
    "eleves_par_ecole": "Élèves par école",
}
POIDS_DEFAUT = {"achev_T": 35, "ecart_tba_achev": 20, "ecoles_perdues_pic": 20,
                "tendance_achev_3a": 15, "eleves_par_ecole": 10}
SENS = {"achev_T": -1, "ecart_tba_achev": 1, "ecoles_perdues_pic": 1,
        "tendance_achev_3a": -1, "eleves_par_ecole": 1}
# Cartogramme : position approximative des 13 régions (ligne, colonne)
GRILLE = {"NORD": (0, 1), "SAHEL": (0, 3), "BOUCLE DU MOUHOUN": (1, 0), "CENTRE-NORD": (1, 2),
          "EST": (1, 4), "CENTRE-OUEST": (2, 1), "CENTRE": (2, 2), "PLATEAU CENTRAL": (2, 3),
          "CENTRE-EST": (2, 4), "HAUTS-BASSINS": (3, 0), "SUD-OUEST": (3, 1), "CENTRE-SUD": (3, 2),
          "CASCADES": (4, 0)}
COULEURS_NIVEAU = {"Critique": "#b71c1c", "Élevé": "#ef6c00", "Modéré": "#f9a825", "Faible": "#2e7d32"}


@st.cache_data
def charger():
    m = pd.read_csv(D / "table_variables.csv").sort_values(["region", "annee_debut"]).reset_index(drop=True)
    g = m.groupby("region")
    m["ecoles_perdues_pic"] = (g.ecoles_total.cummax() - m.ecoles_total) / g.ecoles_total.cummax() * 100
    m["tendance_achev_3a"] = (m.achev_T - g.achev_T.shift(3)) / 3
    prev = pd.read_csv(D / "previsions_2025.csv")
    val = pd.read_csv(D / "validation_modeles.csv")
    sens = pd.read_csv(D / "sensibilite.csv")
    resume = json.loads((D / "resume_chiffres.json").read_text(encoding="utf-8"))
    return m, prev, val, sens, resume


def niveau(v):
    return "Critique" if v >= 65 else "Élevé" if v >= 50 else "Modéré" if v >= 35 else "Faible"


def contributions(df, poids):
    """Contribution de chaque indicateur au score (points sur 100). Somme = indice de risque."""
    tot = sum(poids.values())
    out = pd.DataFrame(index=df.index)
    for c, w in poids.items():
        lo, hi = df[c].min(), df[c].max()
        x = (df[c] - lo) / (hi - lo) if hi > lo else df[c] * 0
        out[c] = (w / tot) * (x if SENS[c] > 0 else 1 - x) * 100
    return out


def sans_accent(t):
    t = unicodedata.normalize("NFKD", str(t)).encode("ascii", "ignore").decode()
    return t.upper().replace("-", " ").replace("'", " ").strip()


def carte(df, col, titre):
    geo = D / "regions_bf.geojson"
    if geo.exists():  # carte géographique réelle si le fond de carte est fourni
        gj = json.loads(geo.read_text(encoding="utf-8"))
        cles = list(gj["features"][0]["properties"].keys())
        cle = next((k for k in ("shapeName", "admin1Name", "NAME_1", "ADM1_FR", "ADM1_EN", "region") if k in cles), cles[0])
        for f in gj["features"]:
            f["properties"]["_k"] = sans_accent(f["properties"][cle])
        d = df.assign(_k=df.region.map(sans_accent))
        fig = px.choropleth(d, geojson=gj, locations="_k", featureidkey="properties._k", color=col,
                            color_continuous_scale="RdYlGn_r", range_color=(0, 100), hover_name="region", title=titre)
        fig.update_geos(fitbounds="locations", visible=False)
        manquantes = set(d._k) - {f["properties"]["_k"] for f in gj["features"]}
        if manquantes:
            st.warning(f"Régions non reconnues dans le fond de carte : {sorted(manquantes)}")
        return fig
    z = [[None] * 5 for _ in range(5)]
    txt = [[""] * 5 for _ in range(5)]
    for _, r in df.iterrows():
        i, j = GRILLE[r.region]
        z[i][j] = r[col]
        txt[i][j] = f"{r.region.title()}<br><b>{r[col]:.0f}</b>"
    fig = go.Figure(go.Heatmap(z=z, text=txt, texttemplate="%{text}", colorscale="RdYlGn_r", zmin=0, zmax=100,
                               xgap=5, ygap=5, hoverinfo="text", colorbar=dict(title="Risque")))
    fig.update_yaxes(autorange="reversed", visible=False)
    fig.update_xaxes(visible=False)
    fig.update_layout(title=titre, height=520, margin=dict(t=60, b=10))
    return fig


m, prev, val, sens, resume = charger()

# --------------------------------------------------------------------------- barre latérale
st.sidebar.title("🎓 EDU-PREDICT BURKINA")
st.sidebar.caption("Aide à la décision : où le risque de décrochage au primaire est-il le plus fort ?")
page = st.sidebar.radio("Navigation", ["Vue d'ensemble", "Carte du risque", "Fiche région",
                                       "Prévision et alertes", "Méthode et limites"])
annee = st.sidebar.slider("Année scolaire (rentrée)", 2018, 2024, 2024,
                          help="2024 = année scolaire 2024/2025")
st.sidebar.markdown("**Poids des indicateurs (%)**")
poids = {c: st.sidebar.slider(LIBELLES[c], 0, 100, POIDS_DEFAUT[c], 5) for c in POIDS_DEFAUT}
if sum(poids.values()) == 0:
    poids = dict(POIDS_DEFAUT)
    st.sidebar.warning("Tous les poids à zéro : valeurs par défaut utilisées.")
st.sidebar.caption("Les poids sont normalisés pour totaliser 100 %. Valeurs par défaut = choix du projet.")

dy = m[m.annee_debut == annee].copy()
contrib = contributions(dy, poids)
dy["indice_risque"] = contrib.sum(axis=1)
dy["rang"] = dy.indice_risque.rank(ascending=False).astype(int)
dy["niveau"] = dy.indice_risque.map(niveau)
dy = dy.sort_values("rang")

# --------------------------------------------------------------------------- pages
if page == "Vue d'ensemble":
    st.title("Risque de décrochage au primaire : vue d'ensemble")
    st.caption(f"Année scolaire {annee}/{annee + 1} · 13 régions · source DGESS")
    ach15 = m[m.annee_debut == 2015].achev_T.mean()
    ach24 = m[m.annee_debut == 2024].achev_T.mean()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Régions en niveau critique ou élevé", int(dy.niveau.isin(["Critique", "Élevé"]).sum()))
    c2.metric("Région la plus à risque", dy.iloc[0].region.title(), f"{dy.iloc[0].indice_risque:.0f}/100", delta_color="off")
    c3.metric("Achèvement moyen des régions, 2024", f"{ach24:.1f} %", f"{ach24 - ach15:+.1f} pts depuis 2015")
    c4.metric("Écoles primaires (13 régions)", f"{int(dy.ecoles_total.sum()):,}".replace(",", " "))
    cg, cd = st.columns(2)
    with cg:
        fig = px.bar(dy.sort_values("indice_risque"), x="indice_risque", y="region", orientation="h",
                     color="niveau", color_discrete_map=COULEURS_NIVEAU, range_x=[0, 100],
                     labels={"indice_risque": "Indice de risque (0-100)", "region": ""},
                     title="Classement des régions")
        st.plotly_chart(fig)
    with cd:
        ev = m.groupby("annee_debut").achev_T.mean().reset_index()
        fig = px.line(ev, x="annee_debut", y="achev_T", markers=True,
                      labels={"annee_debut": "Rentrée", "achev_T": "Achèvement moyen (%)"},
                      title="Achèvement moyen des 13 régions")
        st.plotly_chart(fig)
    st.info("Lecture : un indice élevé signale un risque relatif plus fort **entre régions** ; ce n'est pas une "
            "probabilité individuelle de décrochage. Voir « Méthode et limites ».")

elif page == "Carte du risque":
    st.title("Carte du risque de décrochage")
    st.plotly_chart(carte(dy, "indice_risque", f"Indice de risque {annee}/{annee + 1}"))
    if not (D / "regions_bf.geojson").exists():
        st.caption("Carte schématique (une case par région, position approximative). Pour la carte géographique, "
                   "placer le fond de carte des 13 régions (GeoJSON) dans data/regions_bf.geojson.")
    t = dy[["rang", "region", "indice_risque", "niveau", "achev_T", "ecoles_total"]].copy()
    t.columns = ["Rang", "Région", "Indice", "Niveau", "Achèvement (%)", "Écoles"]
    st.dataframe(t.round(1), hide_index=True)

elif page == "Fiche région":
    st.title("Fiche région")
    reg = st.selectbox("Région", sorted(m.region.unique()), index=sorted(m.region.unique()).index(dy.iloc[0].region))
    r = dy[dy.region == reg].iloc[0]
    p = prev[prev.region == reg].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Indice de risque", f"{r.indice_risque:.0f}/100", f"rang {r['rang']} sur 13", delta_color="off")
    c2.metric("Niveau", r.niveau)
    c3.metric("Achèvement", f"{r.achev_T:.1f} %")
    c4.metric("Prévision 2025/26", f"{p.prevision_achev_2025:.1f} %",
              f"intervalle 80 % : {p.borne_basse_80:.0f} à {p.borne_haute_80:.0f}", delta_color="off")
    st.markdown(f"**Alerte (2024/25) : {p.alerte}**")
    s = m[m.region == reg]
    cg, cd = st.columns(2)
    with cg:
        a = s.melt(id_vars="annee_debut", value_vars=["achev_F", "achev_G", "achev_T"], var_name="groupe", value_name="v")
        a["groupe"] = a.groupe.map({"achev_F": "Filles", "achev_G": "Garçons", "achev_T": "Ensemble"})
        st.plotly_chart(px.line(a, x="annee_debut", y="v", color="groupe", markers=True,
                                labels={"annee_debut": "Rentrée", "v": "Achèvement (%)", "groupe": ""},
                                title="Taux d'achèvement"))
    with cd:
        st.plotly_chart(px.bar(s, x="annee_debut", y="ecoles_total",
                               labels={"annee_debut": "Rentrée", "ecoles_total": "Écoles primaires"},
                               title="Nombre d'écoles primaires"))
    cc = contrib.loc[dy.index[dy.region == reg]].iloc[0].rename(index=LIBELLES).sort_values()
    st.plotly_chart(px.bar(cc, orientation="h", labels={"value": "Points de risque (sur 100)", "index": ""},
                           title=f"Ce qui explique l'indice de {reg.title()} en {annee}").update_layout(showlegend=False))
    principal = cc.idxmax()
    st.markdown(f"**Facteur principal :** {principal.lower()}. "
                "Les pistes d'action sont à valider avec les directions régionales : "
                "suivi des cohortes en fin de cycle, appui aux écoles fermées ou saturées, "
                "attention aux écarts entre filles et garçons.")

elif page == "Prévision et alertes":
    st.title("Prévision 2025/26 et alertes")
    st.write("Prévision du **taux d'achèvement** de l'année suivante. Le modèle retenu est celui qui se trompe "
             "le moins sur les années 2022 à 2024 (validation dans le temps).")
    t = prev[["region", "achev_T", "prevision_achev_2025", "borne_basse_80", "borne_haute_80",
              "indice_risque", "niveau", "alerte"]].copy()
    t.columns = ["Région", "Achèvement 2024/25", "Prévision 2025/26", "Bas (80 %)", "Haut (80 %)",
                 "Indice", "Niveau", "Alerte"]
    st.dataframe(t.round(1), hide_index=True)
    st.download_button("Télécharger les prévisions (CSV)", prev.to_csv(index=False).encode("utf-8"),
                       "previsions_2025.csv", "text/csv")
    fig = go.Figure()
    p2 = prev.sort_values("prevision_achev_2025")
    fig.add_trace(go.Scatter(x=p2.prevision_achev_2025, y=p2.region.str.title(), mode="markers", name="Prévision",
                             error_x=dict(type="data", symmetric=False,
                                          array=p2.borne_haute_80 - p2.prevision_achev_2025,
                                          arrayminus=p2.prevision_achev_2025 - p2.borne_basse_80)))
    fig.update_layout(title="Achèvement prévu et intervalle à 80 %", xaxis_title="Achèvement (%)")
    st.plotly_chart(fig)
    tot = val[val.annee_cible == "2022-2024"][["modele", "MAE", "RMSE", "n"]]
    st.subheader("Comparaison des modèles (validation 2022-2024)")
    st.dataframe(tot.round(2), hide_index=True)
    st.info(f"Meilleur modèle : **{resume['meilleur_modele']}** (erreur absolue moyenne "
            f"{resume['mae_par_modele'][resume['meilleur_modele']]} points). Avec 13 régions et une rupture en 2022, "
            "un modèle simple est plus fiable qu'un modèle complexe ; l'intervalle est asymétrique car les baisses "
            "d'achèvement ont été plus fortes que les hausses.")

else:
    st.title("Méthode et limites")
    st.markdown("""
**Données.** Statistiques DGESS du préscolaire et du primaire (2015/16 à 2024/25), 13 régions.
**Indice de risque (0 à 100).** Moyenne pondérée de 5 indicateurs, chacun normalisé entre régions
(0 = meilleure région, 1 = moins bonne). Plus l'indice est haut, plus le risque relatif est fort.
""")
    st.dataframe(pd.DataFrame({"Indicateur": [LIBELLES[c] for c in POIDS_DEFAUT],
                               "Poids du projet (%)": list(POIDS_DEFAUT.values())}), hide_index=True)
    st.subheader("Robustesse de l'indice (2024/25)")
    st.write(f"Avec 2 000 jeux de poids tirés au hasard, le classement reste proche de celui du projet "
             f"(corrélation de rang moyenne {resume['rho_spearman_poids_aleatoires_moyen']}, "
             f"5e percentile {resume['rho_spearman_poids_aleatoires_p5']}). Les mêmes cinq régions arrivent en tête "
             f"dans {resume['top5_identique_pct_des_tirages']} % des tirages.")
    st.dataframe(sens, hide_index=True)
    st.subheader("Limites")
    st.markdown("""
- Le **décrochage** n'est pas mesuré directement : le taux d'achèvement est un indicateur approché.
  Il ne distingue pas l'abandon de la **fermeture d'écoles** (très marquée au Sahel, à l'Est, au Centre-Nord
  et en Boucle du Mouhoun). L'indice la prend en compte séparément.
- Le résultat est un **indice de vulnérabilité territoriale** par région, pas une probabilité individuelle.
- Seulement 13 régions et 10 années : les prévisions sont prudentes et accompagnées d'un intervalle.
- Les corrélations ne prouvent pas une cause. Le primaire seul est couvert (pas le secondaire).
- Les définitions et unités des indicateurs doivent être confirmées avec la publication d'origine.
""")
