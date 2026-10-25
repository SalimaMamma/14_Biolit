"""
eval_crops.py — Évaluation dans les conditions de production (crops YOLO du test)
===================================================================================
Compare, sur les crops YOLO des photos du split TEST (make_crops.py) :
    - le modèle livré (results/, exécuté par le code de production ONNX)
    - la v2 ré-entraînée sur train + val (même protocole que train.py)

Pré-requis :
    make_crops.py --split test
    extract_features.py --dataset crops --name crops_fp32

Usage :
    uv run python -m ml.classification.training.eval_crops
"""

import json

import numpy as np
import pandas as pd

from ..classifier_infer import load_model, predict_features
from ..config import RESULTS_DIR
from .baseline_v2 import V2Model
from .data import SPLIT_FILE, build_taxonomy, load_data, make_split
from .evaluation import bootstrap, detailed_metrics, groups_of, to_preds
from .extract_features import load_features
from .make_crops import CROPS_FILE
from .train import common_species


def main():
    split_of = pd.read_parquet(SPLIT_FILE).set_index("image_path")["split"]
    crops = pd.read_parquet(CROPS_FILE)
    crops = crops[crops["source_path"].map(split_of) == "test"].reset_index(drop=True)
    feats = load_features("crops_fp32")
    missing = (~crops["image_path"].isin(feats)).sum()
    if missing:
        print(f"  ⚠ {missing} crops sans features : relancer extract_features --dataset crops")
    crops = crops[crops["image_path"].isin(feats)].reset_index(drop=True)
    Xc = np.stack([feats[p] for p in crops["image_path"]])
    print(f"  {len(crops)} crops de test")

    model = load_model(source="local", with_encoder=False)
    preds = {"final": predict_features(Xc, model)}

    df, X = load_data()
    split = make_split(df, X)
    taxonomy, parents = build_taxonomy(df)
    tr, va = split == "train", split == "val"
    v2 = V2Model.fit(df[tr].reset_index(drop=True), X[tr])
    v2.calibrate(X[va], df.loc[va, "species_name"].to_numpy())
    trva = pd.concat([df[tr], df[va]], ignore_index=True)
    v2_final = V2Model.fit(trva, np.concatenate([X[tr], X[va]]))
    v2_final.T_proto = v2.T_proto
    preds["v2_reentrainee"] = to_preds(*v2_final.predict(Xc), taxonomy, parents, "v2")

    common = common_species(trva)
    learned = model.meta["classes"]["species_name"]
    report = {"n_crops": len(crops),
              "resultats": {k: detailed_metrics(p, crops, common, learned) for k, p in preds.items()},
              "bootstrap": {f"final_vs_{k}": bootstrap(crops, preds["final"], p, groups_of(crops, common, learned))
                            for k, p in preds.items() if k != "final"}}
    for k, r in report["resultats"].items():
        a = r["toutes"]
        print(f"  {k:16s} préc={a['precision']:.3f} espèces={a['espece_correctes']} "
              f"préc.espèce={a['espece_precision']:.3f} famille={a['colonnes']['famille']['correcte']:.3f} | "
              f"communes={r['communes']['espece_correctes']} rares={r['rares']['espece_correctes']}")
    with open(RESULTS_DIR / "crops_eval.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
