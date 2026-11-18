"""Fusionne les 7 tableaux propres en une table maître (13 régions x 10 ans) + une table nationale de contrôle."""
import pandas as pd
from pathlib import Path

D = Path(__file__).parent.parent / "data"
tables = {"effectifs": "effectifs_region_sexe", "cp1": "nouveaux_cp1_region_sexe",
          "tba": "tba_region_sexe", "tbs": "tbs_region_sexe",
          "achev": "achevement_region_sexe", "cep": "cep_region_sexe"}
cles = ["region", "annee_debut"]

master = None
for nom, f in tables.items():
    w = pd.read_csv(D / f"{f}.csv").pivot(index=cles, columns="sexe", values="valeur")
    w.columns = [f"{nom}_{c}" for c in w.columns]
    master = w if master is None else master.join(w)

e = pd.read_csv(D / "ecoles_region_statut.csv").pivot(index=cles, columns="statut", values="valeur")
e.columns = [f"ecoles_{c.lower().replace('é', 'e')}" for c in e.columns]
master = master.join(e).reset_index()

national = master[master.region == "BURKINA FASO"]
master = master[master.region != "BURKINA FASO"]
national.to_csv(D / "table_nationale_controle.csv", index=False)
master.to_csv(D / "table_maitre.csv", index=False)
print(master.shape, master.isna().sum().sum(), "valeurs vides")  # attendu : (130, 23) 0
