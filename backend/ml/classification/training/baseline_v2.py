"""
baseline_v2.py — Méthode v2, référence de non-régression
========================================================
v2 = prototypes visuels (moyenne des features whitened par espèce) pour l'espèce, sinon
MLP du niveau le plus fin (famille → règne), seuil 0,35 et marge 0,05 partout ; seule la
température des prototypes est calibrée (MLP de niveau : T = 1), comme en production v2.

V2Model.fit(...) ré-entraîne la v2 sur les mêmes données que le nouveau modèle ; train.py
refuse tout réglage qui ferait moins bien qu'elle.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from ..config import MIN_PROTO_IMAGES, SUPERVISED_LEVELS
from .components import fit_temperature, fit_whitening, l2n, label_index, softmax
from .mlp import mlp_logits, train_mlp

V2_THRESHOLD, V2_MARGIN = 0.35, 0.05
FINE = ["famille", "ordre", "classe", "phylum", "regne"]


def _top2(p):
    idx = np.argsort(-p, axis=1)[:, :2]
    r = np.arange(len(p))
    return idx[:, 0], p[r, idx[:, 0]], p[r, idx[:, 0]] - p[r, idx[:, 1]]


def v2_decide(p_proto, species, level_probs, level_classes):
    """Décision v2 vectorisée → (niveaux, labels)."""
    n = len(p_proto)
    lvl = np.full(n, None, dtype=object)
    lab = np.full(n, None, dtype=object)
    i, s, m = _top2(p_proto)
    sel = (s >= V2_THRESHOLD) & (m >= V2_MARGIN)
    lvl[sel], lab[sel] = "species_name", np.array(species, dtype=object)[i[sel]]
    for L in FINE:
        if L not in level_probs:
            continue
        i, s, m = _top2(level_probs[L])
        sel = np.equal(lvl, None) & (s >= V2_THRESHOLD) & (m >= V2_MARGIN)
        lvl[sel], lab[sel] = L, np.array(level_classes[L], dtype=object)[i[sel]]
    return lvl, lab


@dataclass
class V2Model:
    mean: np.ndarray
    W: np.ndarray
    species: list
    protos: np.ndarray
    level_mlps: dict          # {niveau → (mlp, classes)}
    T_proto: float = 1.0

    @classmethod
    def fit(cls, df: pd.DataFrame, X: np.ndarray) -> "V2Model":
        mean, W = fit_whitening(X)
        Z = l2n((X - mean) @ W)
        sp = df["species_name"].to_numpy(dtype=object)
        counts = pd.Series(sp).value_counts()
        species = sorted(counts[counts >= MIN_PROTO_IMAGES].index)
        protos = l2n(np.stack([Z[sp == s].mean(0) for s in species]))
        mlps = {}
        for lvl in SUPERVISED_LEVELS:
            ok = df[lvl].notna().to_numpy()
            classes = sorted(df.loc[ok, lvl].unique())
            cidx = {c: i for i, c in enumerate(classes)}
            y = np.array([cidx[c] for c in df.loc[ok, lvl]])
            mlps[lvl] = (train_mlp(Z[ok], y, len(classes)), classes)
        return cls(mean, W, species, protos, mlps)

    def _z(self, X):
        return l2n((X - self.mean) @ self.W)

    def calibrate(self, X_val, labels_val):
        ok, y = label_index(self.species, labels_val)
        self.T_proto = 1.0 / fit_temperature(self._z(X_val[ok]) @ self.protos.T, y)

    def predict(self, X):
        Z = self._z(X)
        p_proto = softmax((Z @ self.protos.T) * self.T_proto)
        lp = {L: softmax(mlp_logits(m, Z)) for L, (m, _) in self.level_mlps.items()}
        return v2_decide(p_proto, self.species, lp, {L: c for L, (_, c) in self.level_mlps.items()})
