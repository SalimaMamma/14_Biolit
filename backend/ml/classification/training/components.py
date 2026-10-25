"""
components.py — Composants du modèle « fusion hiérarchique »
============================================================
    prototypes      un vecteur par espèce = α × moyenne visuelle (whitened) + (1 − α) × texte
    régression log. multinomiale sur les features 768d standardisées (scikit-learn)
    fusion          P(espèce) = w × P_LR + (1 − w) × P_proto
    agrégation      P(taxon) = somme des P(espèce) du taxon, pour chaque niveau

FusionModel.probas() (numpy) est la référence ; to_torch() en fait un module exporté en ONNX
(training/export_onnx.py::export_heads), vérifié contre numpy à l'export.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from ..config import (
    BIOCLIP_MODEL_ID,
    DATA_DIR,
    MIN_PROTO_IMAGES,
    SEED,
    SUPERVISED_LEVELS,
    WHITENING_DIM,
)

TEXT_CACHE = DATA_DIR / "text_features.npz"


# ════════════════════════════════════════════════════════════════════════════
# OUTILS
# ════════════════════════════════════════════════════════════════════════════

def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def l2n(x):
    return (x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8, None)).astype(np.float32)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """T > 0 minimisant la NLL de softmax(logits / T) (temperature scaling)."""
    def nll(log_t):
        p = softmax(logits / np.exp(log_t))
        return -np.mean(np.log(p[np.arange(len(y)), y] + 1e-12))
    return float(np.exp(minimize_scalar(nll, bounds=(-5, 7), method="bounded").x))


def nll(p, classes, labels):
    idx = {c: i for i, c in enumerate(classes)}
    lab = np.asarray(labels, dtype=object)
    ok = np.array([c in idx for c in lab])
    y = np.array([idx[c] for c in lab[ok]])
    return float(-np.mean(np.log(p[ok][np.arange(ok.sum()), y] + 1e-12)))


def label_index(classes, labels):
    idx = {c: i for i, c in enumerate(classes)}
    lab = np.asarray(labels, dtype=object)
    ok = np.array([c in idx for c in lab])
    return ok, np.array([idx[c] for c in lab[ok]])


# ════════════════════════════════════════════════════════════════════════════
# TEXTE BIOCLIP-2 (prototypes)
# ════════════════════════════════════════════════════════════════════════════

def text_prompt(species: str, tax: dict) -> str:
    """Prompt taxonomique (format d'entraînement de BioCLIP-2)."""
    ranks = " ".join(t for t in (tax.get(lvl) for lvl in SUPERVISED_LEVELS) if t)
    return f"a photo of {ranks} {species.capitalize()}."


def encode_text(texts: list[str]) -> np.ndarray:
    """Encodeur texte BioCLIP-2 (open_clip / PyTorch, entraînement uniquement) → (N, 768)."""
    import open_clip
    import torch

    model, _, _ = open_clip.create_model_and_transforms(BIOCLIP_MODEL_ID)
    tokenizer = open_clip.get_tokenizer(BIOCLIP_MODEL_ID)
    model.eval()
    with torch.no_grad():
        feats = [model.encode_text(tokenizer(texts[i:i + 64])).float().numpy()
                 for i in range(0, len(texts), 64)]
    return l2n(np.concatenate(feats))


def text_features(species: list[str], taxonomy: dict) -> dict[str, np.ndarray]:
    """Features texte BioCLIP-2 par espèce (cache data/…/text_features.npz)."""
    cache = {}
    if TEXT_CACHE.exists():
        d = np.load(TEXT_CACHE)
        cache = dict(zip(d["species"].tolist(), d["features"]))
    todo = [s for s in species if s not in cache]
    if todo:
        print(f"  Encodage texte BioCLIP-2 de {len(todo)} espèces...")
        cache.update(zip(todo, encode_text([text_prompt(s, taxonomy[s]) for s in todo])))
        keys = sorted(cache)
        np.savez(TEXT_CACHE, species=np.array(keys), features=np.stack([cache[k] for k in keys]))
    return cache


# ════════════════════════════════════════════════════════════════════════════
# RÉGRESSION LOGISTIQUE
# ════════════════════════════════════════════════════════════════════════════

def fit_lr(X, labels, C):
    scaler = StandardScaler().fit(X)
    lr = LogisticRegression(C=C, max_iter=1000, random_state=SEED)
    lr.fit(scaler.transform(X), np.asarray(labels, dtype=object))
    return scaler, lr


def choose_C(X, labels, groups, grid) -> tuple[float, dict]:
    """C par validation croisée groupée par sortie (3 plis, log-loss)."""
    labels = np.asarray(labels, dtype=object)
    scores = {}
    for C in grid:
        losses = []
        for f_tr, f_va in GroupKFold(n_splits=3).split(X, labels, groups):
            scaler, lr = fit_lr(X[f_tr], labels[f_tr], C)
            p = lr.predict_proba(scaler.transform(X[f_va]))
            losses.append(nll(p, list(lr.classes_), labels[f_va]))
        scores[C] = float(np.mean(losses))
        print(f"    C={C}: log-loss CV = {scores[C]:.4f}")
    return min(scores, key=scores.get), scores


# ════════════════════════════════════════════════════════════════════════════
# MODÈLE
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class FusionModel:
    species: list                       # espèces apprises (ordre des colonnes)
    wh_mean: np.ndarray                 # whitening : z = l2n((x − mean) @ W)
    wh_W: np.ndarray
    protos: np.ndarray                  # (S, 256)
    sc_mean: np.ndarray                 # standardisation de la LR
    sc_scale: np.ndarray
    lr_coef: np.ndarray                 # (S, 768)
    lr_bias: np.ndarray                 # (S,)
    agg: dict                           # {niveau → (M (S, n_taxons), taxons)}
    C: float
    alpha: float
    T_proto: float = 1.0                # multiplicatif sur les cosinus
    T_lr: float = 1.0                   # diviseur des logits
    w: float = 0.5                      # poids de la LR dans la fusion

    def proto_logits(self, X):
        return l2n((X - self.wh_mean) @ self.wh_W) @ self.protos.T

    def lr_logits(self, X):
        return ((X - self.sc_mean) / self.sc_scale) @ self.lr_coef.T + self.lr_bias

    def species_probas(self, X):
        p_lr = softmax(self.lr_logits(X) / self.T_lr)
        p_pr = softmax(self.proto_logits(X) * self.T_proto)
        return self.w * p_lr + (1 - self.w) * p_pr

    def probas(self, X) -> dict:
        """{"species_name": (N, S), "famille": (N, F), …} — identique au graphe ONNX."""
        p = self.species_probas(X)
        return {"species_name": p, **{lvl: p @ M for lvl, (M, _) in self.agg.items()}}

    def classes(self) -> dict:
        return {"species_name": list(self.species), **{lvl: t for lvl, (_, t) in self.agg.items()}}

    def calibrate(self, X_val, labels_val):
        ok, y = label_index(self.species, labels_val)
        self.T_lr = fit_temperature(self.lr_logits(X_val[ok]), y)
        self.T_proto = 1.0 / fit_temperature(self.proto_logits(X_val[ok]), y)

    def to_torch(self):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        model = self

        class FusionHeads(nn.Module):
            def __init__(self):
                super().__init__()
                t = lambda a: torch.tensor(np.asarray(a, dtype=np.float32))  # noqa: E731
                self.register_buffer("wh_mean", t(model.wh_mean))
                self.register_buffer("wh_W", t(model.wh_W))
                self.register_buffer("protos", t(model.protos))
                self.register_buffer("sc_mean", t(model.sc_mean))
                self.register_buffer("sc_scale", t(model.sc_scale))
                self.register_buffer("lr_coef", t(model.lr_coef))
                self.register_buffer("lr_bias", t(model.lr_bias))
                self.levels = list(model.agg)
                for lvl in self.levels:
                    self.register_buffer(f"M_{lvl}", t(model.agg[lvl][0]))
                self.output_names = ["p_species_name"] + [f"p_{lvl}" for lvl in self.levels]

            def forward(self, x):
                z = F.normalize((x - self.wh_mean) @ self.wh_W, dim=-1)
                p_pr = F.softmax((z @ self.protos.T) * model.T_proto, -1)
                p_lr = F.softmax((((x - self.sc_mean) / self.sc_scale) @ self.lr_coef.T
                                  + self.lr_bias) / model.T_lr, -1)
                p = model.w * p_lr + (1 - model.w) * p_pr
                return tuple([p] + [p @ getattr(self, f"M_{lvl}") for lvl in self.levels])

        return FusionHeads().eval()


def aggregation(species, taxonomy) -> dict:
    agg = {}
    for lvl in SUPERVISED_LEVELS:
        taxa = sorted({taxonomy[s][lvl] for s in species if taxonomy[s][lvl]})
        idx = {t: j for j, t in enumerate(taxa)}
        M = np.zeros((len(species), len(taxa)), dtype=np.float32)
        for i, s in enumerate(species):
            if taxonomy[s][lvl]:
                M[i, idx[taxonomy[s][lvl]]] = 1.0
        agg[lvl] = (M, taxa)
    return agg


def fit_whitening(X):
    pca = PCA(n_components=WHITENING_DIM, whiten=True, random_state=SEED).fit(X)
    W = (pca.components_ / np.sqrt(pca.explained_variance_[:, None] + 1e-8)).T
    return pca.mean_.astype(np.float32), W.astype(np.float32)


def fit_model(df: pd.DataFrame, X: np.ndarray, taxonomy: dict, text: dict, C: float,
              alpha: float) -> FusionModel:
    """Ajuste whitening, prototypes et régression logistique (températures/w : à part)."""
    sp = df["species_name"].to_numpy(dtype=object)
    counts = pd.Series(sp).value_counts()
    species = sorted(counts[counts >= MIN_PROTO_IMAGES].index)
    keep = np.isin(sp, species)

    mean, W = fit_whitening(X)
    Z = l2n((X - mean) @ W)
    vis = l2n(np.stack([Z[sp == s].mean(0) for s in species]))
    txt = l2n((np.stack([text[s] for s in species]) - mean) @ W)
    protos = l2n(alpha * vis + (1 - alpha) * txt)

    scaler, lr = fit_lr(X[keep], sp[keep], C)
    assert list(lr.classes_) == species
    return FusionModel(species=species, wh_mean=mean, wh_W=W, protos=protos,
                       sc_mean=scaler.mean_.astype(np.float32),
                       sc_scale=scaler.scale_.astype(np.float32),
                       lr_coef=lr.coef_.astype(np.float32), lr_bias=lr.intercept_.astype(np.float32),
                       agg=aggregation(species, taxonomy), C=C, alpha=alpha)
