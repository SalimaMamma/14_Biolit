"""
Non-régression des performances sur le jeu de TEST (jamais vu, sans quasi-doublons).

Nécessite les sorties de training/train.py dans results/ :
    eval_features_test.npz, test_predictions.parquet, training_report.json
    (+ classifier_heads.onnx, classifier_meta.json)

Vérifie :
    1. le modèle livré, exécuté par le code de production, reproduit exactement les
       métriques du rapport d'entraînement (déterminisme) ;
    2. il ne dégrade pas la v2 ré-entraînée sur les mêmes données : précision ≥, taux d'erreur ≤,
       espèces correctes ≥, colonne famille correcte ≥.

    RUN_ONNX_REGRESSION=1 uv run pytest ml/classification/tests/test_model_performance.py -vv
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

from ml.classification.config import HEADS_ONNX_FILE, META_FILE, RESULTS_DIR

FEATS = RESULTS_DIR / "eval_features_test.npz"
TRUTH = RESULTS_DIR / "test_predictions.parquet"
REPORT = RESULTS_DIR / "training_report.json"
BASELINES = ["v2_reentrainee"]

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_ONNX_REGRESSION") != "1"
    or not all(p.exists() for p in [FEATS, TRUTH, REPORT, HEADS_ONNX_FILE, META_FILE]),
    reason="RUN_ONNX_REGRESSION=1 et sorties de training/train.py requises",
)


@pytest.fixture(scope="module")
def ctx():
    from ml.classification.classifier_infer import load_model

    with open(REPORT, encoding="utf-8") as f:
        report = json.load(f)
    d = np.load(FEATS, allow_pickle=False)
    truth = pd.read_parquet(TRUTH).set_index("image_path").loc[d["image_path"]].reset_index()
    model = load_model(source="local", with_encoder=False)
    return report, d["features"], truth, model


def test_runtime_reproduces_training_report(ctx):
    from ml.classification.classifier_infer import predict_features
    from ml.classification.training.evaluation import detailed_metrics

    report, X, truth, model = ctx
    got = detailed_metrics(predict_features(X, model), truth, report["especes_communes"],
                           model.meta["classes"]["species_name"])["toutes"]
    expected = report["test"]["final"]["toutes"]
    for k, v in expected.items():
        assert got[k] == (v if isinstance(v, dict) else pytest.approx(v)), k


@pytest.mark.parametrize("baseline", BASELINES)
def test_not_worse_than_v2(ctx, baseline):
    report, *_ = ctx
    a, b = report["test"]["final"]["toutes"], report["test"][baseline]["toutes"]
    assert a["precision"] >= b["precision"] - 1e-9, "précision"
    assert a["taux_erreur"] <= b["taux_erreur"] + 1e-9, "taux d'erreur"
    assert a["espece_correctes"] >= b["espece_correctes"], "espèces correctes"
    assert a["colonnes"]["famille"]["correcte"] >= b["colonnes"]["famille"]["correcte"] - 1e-9, "famille"
