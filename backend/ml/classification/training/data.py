"""
data.py — Données d'entraînement : features, découpage, taxonomie
=================================================================
Découpage (split.parquet) :
    1. train / val / test = 70 / 15 / 15 % des photos, groupé par SORTIE (ID - N1) :
       les photos d'une même sortie restent ensemble.
    2. Purge des quasi-doublons : une photo de val (resp. test) dont une photo de train
       (resp. train + val) a une similarité cosinus ≥ DUPLICATE_COS est marquée « exclu »
       (sa jumelle reste à l'entraînement). Sans cette purge, ~17 % du test avait une
       quasi-jumelle à l'entraînement, ce qui gonflait les scores de 2 à 4 points.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from ..config import DATA_DIR, DUPLICATE_COS, SEED, SUPERVISED_LEVELS
from .build_dataset import DATASET_FILE
from .extract_features import load_features

SPLIT_FILE = DATA_DIR / "split.parquet"


def load_data(feat_name: str = "fp32") -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_parquet(DATASET_FILE)
    feats = load_features(feat_name)
    df = df[df["image_path"].isin(feats)].reset_index(drop=True)
    X = np.stack([feats[p] for p in df["image_path"]]).astype(np.float32)
    return df, X


def _max_cos(A: np.ndarray, B: np.ndarray, chunk: int = 1024) -> np.ndarray:
    return np.concatenate([(A[i:i + chunk] @ B.T).max(1) for i in range(0, len(A), chunk)])


def make_split(df: pd.DataFrame, X: np.ndarray) -> np.ndarray:
    """→ tableau de « train » | « val » | « test » | « exclu » aligné sur df."""
    if SPLIT_FILE.exists():
        saved = pd.read_parquet(SPLIT_FILE).set_index("image_path")["split"]
        if set(df["image_path"]) <= set(saved.index):
            return df["image_path"].map(saved).to_numpy()

    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    trval, test = next(gss.split(df, groups=df["id_n1"]))
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15 / 0.85, random_state=SEED)
    tr, va = next(gss.split(trval, groups=df["id_n1"].iloc[trval]))
    split = np.empty(len(df), dtype=object)
    split[test], split[trval[tr]], split[trval[va]] = "test", "train", "val"

    for name, refs in [("val", ["train"]), ("test", ["train", "val"])]:
        idx = np.flatnonzero(split == name)
        dup = _max_cos(X[idx], X[np.isin(split, refs)]) >= DUPLICATE_COS
        split[idx[dup]] = "exclu"
        print(f"  {name} : {dup.sum()} photos exclues (quasi-jumelle dans {'+'.join(refs)})")

    pd.DataFrame({"image_path": df["image_path"], "split": split}).to_parquet(SPLIT_FILE)
    return split


def _clean(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)


def build_taxonomy(df: pd.DataFrame) -> tuple[dict, dict]:
    """taxonomy {espèce → {niveau → taxon}}, taxon_parents {niveau → {taxon → parents}}."""
    first = df.drop_duplicates("species_name")
    taxonomy = {r["species_name"]: {lvl: _clean(r[lvl]) for lvl in SUPERVISED_LEVELS}
                for _, r in first.iterrows()}
    parents = {lvl: {} for lvl in SUPERVISED_LEVELS}
    for tax in taxonomy.values():
        for i, lvl in enumerate(SUPERVISED_LEVELS):
            if tax[lvl] is not None and tax[lvl] not in parents[lvl]:
                parents[lvl][tax[lvl]] = {p: tax[p] for p in SUPERVISED_LEVELS[:i]}
    return taxonomy, parents
