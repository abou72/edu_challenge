"""Variables dérivées (parité, élèves par école, variations, chocs d'écoles) -> table_variables.csv."""
import pandas as pd
from pathlib import Path

D = Path(__file__).parent.parent / "data"
m = pd.read_csv(D / "table_maitre.csv").sort_values(["region", "annee_debut"])
g = m.groupby("region")

m["ips_tba"] = m.tba_F / m.tba_G
m["ips_achev"] = m.achev_F / m.achev_G
m["ecart_fg_achev"] = m.achev_F - m.achev_G
m["eleves_par_ecole"] = m.effectifs_T / m.ecoles_total
m["part_prive"] = m.ecoles_prive / m.ecoles_total
m["var_ecoles_pct"] = g.ecoles_total.pct_change() * 100
m["var_effectifs_pct"] = g.effectifs_T.pct_change() * 100
m["var_achev"] = g.achev_T.diff()
m["ecart_tba_achev"] = m.tba_T - m.achev_T
m["choc_ecoles"] = m.var_ecoles_pct < -10

m.to_csv(D / "table_variables.csv", index=False)
print(m.shape)  # attendu : (130, 33)
