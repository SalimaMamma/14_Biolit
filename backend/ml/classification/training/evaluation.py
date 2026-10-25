"""
evaluation.py — Décision hiérarchique vectorisée, métriques, contraintes, bootstrap
====================================================================================
La décision vectorisée (decide_arrays) reproduit classifier_infer.decide() ; train.py
vérifie l'égalité des deux sur le test.
"""

import itertools

import numpy as np
import pandas as pd

from ..config import DECISION_LEVELS, SUPERVISED_LEVELS

# Poids d'une bonne réponse selon sa finesse (objectif de calibration des seuils)
LEVEL_WEIGHT = {"species_name": 1.0, "famille": 0.6, "ordre": 0.4,
                "classe": 0.3, "phylum": 0.2, "regne": 0.1}
T_SPECIES = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
T_HIGH = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
N_BOOT = 2000


# ════════════════════════════════════════════════════════════════════════════
# DÉCISION
# ════════════════════════════════════════════════════════════════════════════

def top1(probs: dict, classes: dict) -> dict:
    """{niveau → (labels top-1, probabilités top-1)}."""
    out = {}
    for lvl in DECISION_LEVELS:
        i = probs[lvl].argmax(1)
        out[lvl] = (np.array(classes[lvl], dtype=object)[i], probs[lvl][np.arange(len(i)), i])
    return out


def decide_arrays(tops: dict, th: dict):
    """Niveau le plus fin dont la probabilité top-1 atteint son seuil → (niveaux, labels)."""
    n = len(tops["species_name"][0])
    lvl = np.full(n, None, dtype=object)
    lab = np.full(n, None, dtype=object)
    for L in DECISION_LEVELS:
        labels, score = tops[L]
        sel = np.equal(lvl, None) & (score >= th[L])
        lvl[sel], lab[sel] = L, labels[sel]
    return lvl, lab


def to_preds(lvl, lab, taxonomy, parents, path="hier") -> list[dict]:
    """Tableaux (niveau, label) → liste de dicts au format de classifier_infer.decide()."""
    out = []
    for L, b in zip(lvl, lab):
        tax = {k: None for k in SUPERVISED_LEVELS + ["species_name"]}
        if L == "species_name":
            tax.update(taxonomy[b])
            tax["species_name"] = b
        elif L is not None:
            tax.update(parents[L].get(b, {}))
            tax[L] = b
        out.append({"best_level": L, "best_label": b, "path": f"{path}_{L or 'rejet'}", **tax})
    return out


# ════════════════════════════════════════════════════════════════════════════
# SCORE (objectif + contraintes)
# ════════════════════════════════════════════════════════════════════════════

def score_arrays(lvl, lab, df: pd.DataFrame, taxonomy: dict, groups: dict) -> dict:
    truth = {L: df[L].astype(str).to_numpy() for L in DECISION_LEVELS}
    pred = ~np.equal(lvl, None)
    ok = np.array([L is not None and str(b) == truth[L][i] for i, (L, b) in enumerate(zip(lvl, lab))])
    sp = lvl == "species_name"
    fam = np.array([taxonomy[b]["famille"] if L == "species_name" else (b if L == "famille" else None)
                    for L, b in zip(lvl, lab)], dtype=object)
    fam_filled = ~np.equal(fam, None)
    fam_ok = fam_filled & (fam.astype(str) == truth["famille"])
    s = {"n": len(df), "n_pred": int(pred.sum()), "n_ok": int(ok.sum()),
         "precision": float(ok.sum() / max(pred.sum(), 1)),
         "n_species": int(sp.sum()), "ok_species": int((ok & sp).sum()),
         "precision_species": float((ok & sp).sum() / max(sp.sum(), 1)),
         "weighted": float(sum(LEVEL_WEIGHT[L] * (ok & (lvl == L)).sum() for L in DECISION_LEVELS)),
         "famille_correctes": int(fam_ok.sum()),
         "precision_famille": float(fam_ok.sum() / max(fam_filled.sum(), 1))}
    s["groupes"] = {g: {"ok_species": int((ok & sp & m).sum()), "n_species": int((sp & m).sum()),
                        "n_ok": int((ok & m).sum()), "n_pred": int((pred & m).sum())}
                    for g, m in groups.items()}
    return s


def admissible(s: dict, ref: dict) -> bool:
    """
    Non-régression vs la référence (v2 ré-entraînée) :
      - précisions globale / espèce / famille et nombre de familles correctes ≥ référence ;
      - dans chaque sous-groupe (espèces communes, rares) : espèces correctes,
        précision et précision espèce ≥ référence.
    """
    if not (s["precision"] >= ref["precision"] and s["precision_species"] >= ref["precision_species"]
            and s["precision_famille"] >= ref["precision_famille"]
            and s["famille_correctes"] >= ref["famille_correctes"]):
        return False
    for g, r in ref["groupes"].items():
        a = s["groupes"][g]
        if (a["ok_species"] < r["ok_species"]
                or a["n_ok"] * max(r["n_pred"], 1) < r["n_ok"] * max(a["n_pred"], 1)
                or a["ok_species"] * max(r["n_species"], 1) < r["ok_species"] * max(a["n_species"], 1)):
            return False
    return True


def failed_constraints(s: dict, ref: dict) -> list:
    out = [k for k in ["precision", "precision_species", "precision_famille", "famille_correctes"]
           if s[k] < ref[k]]
    for g, r in ref["groupes"].items():
        a = s["groupes"][g]
        if a["ok_species"] < r["ok_species"]:
            out.append(f"{g}.especes_correctes")
        if a["n_ok"] * max(r["n_pred"], 1) < r["n_ok"] * max(a["n_pred"], 1):
            out.append(f"{g}.precision")
        if a["ok_species"] * max(r["n_species"], 1) < r["ok_species"] * max(a["n_species"], 1):
            out.append(f"{g}.precision_espece")
    return out


def tune_thresholds(tops, df, taxonomy, groups, ref, strict=True):
    """Seuils (espèce, famille, ordre, + un commun classe/phylum/règne) maximisant le score
    pondéré sous les contraintes (strict) ou sous précision ≥ référence (relâché)."""
    best = None
    for t_sp, t_fam, t_ord, t_hi in itertools.product(T_SPECIES, T_SPECIES, T_SPECIES, T_HIGH):
        th = {"species_name": t_sp, "famille": t_fam, "ordre": t_ord,
              "classe": t_hi, "phylum": t_hi, "regne": t_hi}
        s = score_arrays(*decide_arrays(tops, th), df, taxonomy, groups)
        ok = admissible(s, ref) if strict else s["precision"] >= ref["precision"]
        if ok and (best is None or (s["weighted"], s["precision"]) > (best[1]["weighted"], best[1]["precision"])):
            best = (th, s)
    return best or (None, None)


# ════════════════════════════════════════════════════════════════════════════
# MÉTRIQUES DÉTAILLÉES ET BOOTSTRAP
# ════════════════════════════════════════════════════════════════════════════

def groups_of(df, common, proto_species) -> dict:
    sp = df["species_name"].to_numpy()
    return {"toutes": np.ones(len(df), bool), "communes": np.isin(sp, common),
            "rares": np.isin(sp, proto_species) & ~np.isin(sp, common),
            "inconnues": ~np.isin(sp, proto_species)}


def _correct(preds, df):
    lvl = [p["best_level"] for p in preds]
    rows = df.to_dict("records")
    ok = np.array([lv is not None and str(p["best_label"]) == str(r[lv])
                   for p, lv, r in zip(preds, lvl, rows)])
    sp = np.array([lv == "species_name" for lv in lvl])
    fam = np.array([p.get("famille") is not None and str(p.get("famille")) == str(r["famille"])
                    for p, r in zip(preds, rows)])
    pred = np.array([lv is not None for lv in lvl])
    return pred, ok, sp, fam


def detailed_metrics(preds: list[dict], df: pd.DataFrame, common, proto_species) -> dict:
    pred, ok, is_sp, _ = _correct(preds, df)
    lvl = np.array([p["best_level"] or "rejected" for p in preds], dtype=object)
    out = {}
    for name, g in groups_of(df, common, proto_species).items():
        n = int(g.sum())
        gp, gs = g & pred, g & is_sp
        res = {"n": n, "rejet": float((g & ~pred).sum() / max(n, 1)),
               "precision": float((ok & gp).sum() / max(gp.sum(), 1)),
               "erreurs": int((gp & ~ok).sum()), "taux_erreur": float((gp & ~ok).sum() / max(n, 1)),
               "espece_couverture": float(gs.sum() / max(n, 1)),
               "espece_precision": float((ok & gs).sum() / max(gs.sum(), 1)),
               "espece_correctes": int((ok & gs).sum()),
               "score_pondere": float(sum(LEVEL_WEIGHT[k] * (g & ok & (lvl == k)).sum()
                                          for k in LEVEL_WEIGHT) / max(n, 1)),
               "repartition": {k: int((g & (lvl == k)).sum()) for k in DECISION_LEVELS + ["rejected"]},
               "colonnes": {}}
        for col in SUPERVISED_LEVELS + ["species_name"]:
            pc = np.array([p.get(col) for p in preds], dtype=object)
            filled = g & ~np.equal(pc, None)
            correct = filled & (pc.astype(str) == df[col].astype(str).to_numpy())
            res["colonnes"][col] = {"remplie": float(filled.sum() / max(n, 1)),
                                    "correcte": float(correct.sum() / max(n, 1)),
                                    "precision": float(correct.sum() / max(filled.sum(), 1))}
        out[name] = res
    out["chemins"] = {k: int(v) for k, v in pd.Series([p["path"] for p in preds]).value_counts().items()}
    return out


def bootstrap(df, preds_a, preds_b, groups, seed=0) -> dict:
    """Écarts a − b en points, IC 95 %, bootstrap apparié par sortie (ID - N1)."""
    A, B = _correct(preds_a, df), _correct(preds_b, df)
    clusters = df["id_n1"].to_numpy()
    uniq = np.unique(clusters)
    by = {c: np.flatnonzero(clusters == c) for c in uniq}
    rng = np.random.default_rng(seed)

    def m(stats, idx, g):
        pred, ok, sp, fam = (s[idx] for s in stats)
        g = g[idx]
        return {"precision": (ok & g).sum() / max((pred & g).sum(), 1),
                "espece_correctes": (ok & sp & g).sum() / max(g.sum(), 1),
                "famille_correcte": (fam & g).sum() / max(g.sum(), 1)}

    out, full = {}, np.arange(len(df))
    draws = [np.concatenate([by[c] for c in rng.choice(uniq, len(uniq))]) for _ in range(N_BOOT)]
    for gname, g in groups.items():
        pa, pb = m(A, full, g), m(B, full, g)
        boots = [{k: a - b for (k, a), b in zip(m(A, idx, g).items(), m(B, idx, g).values())}
                 for idx in draws]
        out[gname] = {k: {"ecart": round(float(pa[k] - pb[k]) * 100, 2),
                          "ic95": [round(float(np.percentile([b[k] for b in boots], q)) * 100, 2)
                                   for q in (2.5, 97.5)]}
                      for k in pa}
    return out
