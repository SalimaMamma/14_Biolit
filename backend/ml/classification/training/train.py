"""
train.py — Entraînement, calibration, export et évaluation du modèle « fusion hiérarchique »
=============================================================================================
Pré-requis :
    build_dataset.py → export_onnx.py → extract_features.py   (voir README)

Étapes :
    1. Découpage train / val / test groupé par sortie + purge des quasi-doublons (data.py).
    2. Référence de non-régression : la méthode v2 ré-entraînée sur train (baseline_v2.py).
    3. Modèle sur train : whitening, prototypes (α), régression logistique (C par CV groupée).
       Sur val : températures, poids de fusion w et seuils par niveau, choisis pour maximiser
       les bonnes réponses pondérées par finesse SOUS CONTRAINTE de faire au moins aussi bien
       que la v2 (précisions, familles correctes, sous-groupes communes / rares).
    4. Ré-entraînement sur train + val (C, α, T, w, seuils conservés), export ONNX des têtes,
       classifier_meta.json.
    5. Évaluation sur test (jamais vu, sans quasi-doublons) via le code de production, contre
       la v2 ré-entraînée sur train + val.

Sorties (results/) : classifier_heads.onnx, classifier_meta.json, training_report.json,
                     test_predictions.parquet, eval_features_test.npz

Usage :
    uv run python -m ml.classification.training.train
"""

import datetime
import json

import numpy as np
import pandas as pd

from ..classifier_infer import load_model, predict_features, predict_proba
from ..config import (
    BIOCLIP_MODEL_ID,
    DECISION_LEVELS,
    ENCODER_ONNX_FILE,
    FEATURE_DIM,
    FUSION_WEIGHTS,
    HEADS_ONNX_FILE,
    LR_C_GRID,
    META_FILE,
    MIN_PROTO_IMAGES,
    N_COMMON_SPECIES,
    PROTO_ALPHA,
    RESULTS_DIR,
    SUPERVISED_LEVELS,
    WHITENING_DIM,
)
from .baseline_v2 import V2Model
from .components import choose_C, fit_model, text_features
from .data import build_taxonomy, load_data, make_split
from .evaluation import (
    bootstrap,
    decide_arrays,
    detailed_metrics,
    failed_constraints,
    groups_of,
    score_arrays,
    to_preds,
    top1,
    tune_thresholds,
)
from .export_onnx import export_heads

REPORT_FILE = RESULTS_DIR / "training_report.json"
VERSION = "v3-fusion-hierarchique"


def common_species(df: pd.DataFrame) -> list:
    counts = df["species_name"].value_counts()
    return counts.sort_index().sort_values(ascending=False, kind="stable").index[:N_COMMON_SPECIES].tolist()


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {"date": datetime.datetime.now().isoformat(timespec="seconds"), "version": VERSION}

    # ── 1. données ──
    df, X = load_data()
    split = make_split(df, X)
    taxonomy, parents = build_taxonomy(df)
    part = {s: (df[split == s].reset_index(drop=True), X[split == s]) for s in ["train", "val", "test"]}
    (tr, Xtr), (va, Xva), (te, Xte) = part["train"], part["val"], part["test"]
    report["data"] = {s: {"images": int((split == s).sum()),
                          "especes": int(df.loc[split == s, "species_name"].nunique()),
                          "sorties": int(df.loc[split == s, "id_n1"].nunique())}
                      for s in ["train", "val", "test", "exclu"]}
    print(json.dumps(report["data"]))
    text = text_features(sorted(df["species_name"].unique()), taxonomy)

    common_tr = common_species(tr)
    learned_tr = sorted(tr["species_name"].value_counts().loc[lambda c: c >= MIN_PROTO_IMAGES].index)
    g_val = {k: v for k, v in groups_of(va, common_tr, learned_tr).items() if k in ("communes", "rares")}

    # ── 2. référence v2 (train → val) ──
    print("\n=== Référence v2 (train → val) ===")
    v2 = V2Model.fit(tr, Xtr)
    v2.calibrate(Xva, va["species_name"].to_numpy())
    ref = score_arrays(*v2.predict(Xva), va, taxonomy, g_val)
    report["val"] = {"reference_v2": ref}
    print(f"  v2 : préc={ref['precision']:.3f} espèces={ref['ok_species']} familles={ref['famille_correctes']}")

    # ── 3. modèle (train) + réglages (val) ──
    print("\n=== Modèle sur train, réglages sur val ===")
    sp_tr = tr["species_name"].to_numpy(dtype=object)
    keep = np.isin(sp_tr, learned_tr)
    C, cv = choose_C(Xtr[keep], sp_tr[keep], tr["id_n1"].to_numpy()[keep], LR_C_GRID)
    model = fit_model(tr, Xtr, taxonomy, text, C, PROTO_ALPHA)
    model.calibrate(Xva, va["species_name"].to_numpy())
    print(f"  C={C} | T_lr={model.T_lr:.3f} | T_proto={model.T_proto:.2f}")

    candidates = []
    for w in FUSION_WEIGHTS:
        model.w = w
        tops = top1(model.probas(Xva), model.classes())
        th, s = tune_thresholds(tops, va, taxonomy, g_val, ref, strict=True)
        relaxed = th is None
        if relaxed:
            th, s = tune_thresholds(tops, va, taxonomy, g_val, ref, strict=False)
        cand = {"w": w, "seuils": th, "score": s, "relache": relaxed,
                "contraintes_non_tenues": failed_constraints(s, ref)}
        candidates.append(cand)
        print(f"  w={w}: {'RELÂCHÉ' if relaxed else 'strict '} pondéré={s['weighted']:.1f} "
              f"préc={s['precision']:.3f} espèces={s['ok_species']} familles={s['famille_correctes']} | {th}")
    best = max(candidates, key=lambda c: (not c["relache"], c["score"]["weighted"]))
    if best["relache"]:
        print("  ⚠ Aucun réglage ne respecte toutes les contraintes de non-régression : "
              f"réglage relâché retenu (non tenues : {best['contraintes_non_tenues']})")
    report["val"].update({"C": C, "cv_lr": cv, "T_lr": model.T_lr, "T_proto": model.T_proto,
                          "candidats": candidates, "retenu": best})

    # ── 4. ré-entraînement final (train + val) + export ──
    print("\n=== Ré-entraînement final (train + val) ===")
    trva = pd.concat([tr, va], ignore_index=True)
    Xtrva = np.concatenate([Xtr, Xva])
    final = fit_model(trva, Xtrva, taxonomy, text, C, PROTO_ALPHA)
    final.T_lr, final.T_proto, final.w = model.T_lr, model.T_proto, best["w"]
    export_heads(final.to_torch(), HEADS_ONNX_FILE, FEATURE_DIM)

    classes = final.classes()
    meta = {
        "version": VERSION,
        "created": datetime.date.today().isoformat(),
        "bioclip_model_id": BIOCLIP_MODEL_ID,
        "encoder_file": ENCODER_ONNX_FILE.name,
        "heads_file": HEADS_ONNX_FILE.name,
        "feature_dim": FEATURE_DIM,
        "decision": "niveau le plus fin dont la probabilité atteint le seuil (espèce → règne)",
        "thresholds": best["seuils"],
        "classes": classes,
        "taxonomy": {s: taxonomy[s] for s in final.species},
        "taxon_parents": parents,
        "fusion": {"w_lr": final.w, "alpha_proto": PROTO_ALPHA, "C_lr": C, "T_lr": final.T_lr,
                   "T_proto": final.T_proto, "whitening_dim": WHITENING_DIM,
                   "min_photos_par_espece": MIN_PROTO_IMAGES},
        "training_data": report["data"],
    }
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"  Méta-données → {META_FILE}")

    # ── 5. évaluation sur test, via le code de production ──
    print("\n=== Évaluation sur test ===")
    runtime = load_model(source="local", with_encoder=False)
    preds = predict_features(Xte, runtime)
    p_onnx, p_np = predict_proba(Xte, runtime), final.probas(Xte)
    report["onnx_vs_numpy_max_abs"] = {k: float(np.abs(p_onnx[k] - p_np[k]).max()) for k in DECISION_LEVELS}
    lvl, lab = decide_arrays(top1(p_onnx, classes), best["seuils"])
    assert [p["best_level"] for p in preds] == list(lvl), "decide() ≠ décision vectorisée"
    assert [p["best_label"] for p in preds] == list(lab), "decide() ≠ décision vectorisée"

    v2_final = V2Model.fit(trva, Xtrva)
    v2_final.T_proto = v2.T_proto
    preds_v2 = to_preds(*v2_final.predict(Xte), taxonomy, parents, "v2")

    common_f = common_species(trva)
    report["especes_communes"] = common_f
    groups = groups_of(te, common_f, final.species)
    all_preds = {"final": preds, "v2_reentrainee": preds_v2}
    report["test"] = {k: detailed_metrics(p, te, common_f, final.species) for k, p in all_preds.items()}
    report["bootstrap_test"] = {f"final_vs_{k}": bootstrap(te, preds, p, groups)
                                for k, p in all_preds.items() if k != "final"}
    for k, r in report["test"].items():
        a = r["toutes"]
        print(f"  {k:16s} préc={a['precision']:.3f} espèces={a['espece_correctes']} "
              f"({a['espece_correctes'] / a['n']:.3f}) préc.espèce={a['espece_precision']:.3f} "
              f"famille={a['colonnes']['famille']['correcte']:.3f} | communes={r['communes']['espece_correctes']} "
              f"rares={r['rares']['espece_correctes']}")

    out = te[["image_path", "id_observation", "id_n1", "species_name"] + SUPERVISED_LEVELS].copy()
    for k, ps in [("final", preds), ("v2", preds_v2)]:
        out[f"{k}_level"] = [p["best_level"] for p in ps]
        out[f"{k}_label"] = [p["best_label"] for p in ps]
        out[f"{k}_path"] = [p["path"] for p in ps]
    out["final_score"] = [p["best_score"] for p in preds]
    out.to_parquet(RESULTS_DIR / "test_predictions.parquet", index=False)
    np.savez(RESULTS_DIR / "eval_features_test.npz", image_path=te["image_path"].to_numpy().astype(str),
             features=Xte)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=float)
    print(f"\n  Rapport → {REPORT_FILE}")


if __name__ == "__main__":
    main()
