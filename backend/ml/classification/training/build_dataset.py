"""
build_dataset.py — Reconstruction du jeu d'entraînement depuis l'export Biolit
================================================================================
1. Lit l'export Biolit (CSV « export_biolit.csv ») : une ligne par observation,
   avec le nom scientifique et la/les URL(s) des photos (séparées par « | »).
2. Garde les observations avec un nom scientifique, au moins une photo, et non
   marquées « non-identifiable ».
3. Joint la taxonomie TAXREF (même source que biolit/taxref.py).
4. Télécharge les photos (redimensionnées : plus grand côté ≤ 640 px, JPEG q95).

Sortie : data/dataset_biolit/dataset.parquet, une ligne par image :
    image_path, url, id_observation, id_n1, species_name,
    regne, phylum, classe, ordre, famille

Usage :
    uv run python -m ml.classification.training.build_dataset
    uv run python -m ml.classification.training.build_dataset --export chemin.csv --workers 8
"""

import argparse
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from PIL import Image

from ..config import DATA_DIR, IMAGES_DIR, SUPERVISED_LEVELS

EXPORT_FILE = DATA_DIR / "export_biolit.csv"
DATASET_FILE = DATA_DIR / "dataset.parquet"
MAX_SIDE = 640
USER_AGENT = "Biolit-DataForGood-classification/1.0"


# ════════════════════════════════════════════════════════════════════════════
# TAXONOMIE
# ════════════════════════════════════════════════════════════════════════════

def load_taxref() -> pd.DataFrame:
    """Retourne TAXREF (species_name en minuscules + niveaux) via biolit.taxref."""
    from biolit import DATADIR
    from biolit.taxref import format_taxref

    path = DATADIR / "taxref.parquet"
    if not path.exists():
        print("  TAXREF absent → téléchargement / formatage via biolit.taxref")
        DATADIR.mkdir(parents=True, exist_ok=True)
        format_taxref()
    return pd.read_parquet(path)


# ════════════════════════════════════════════════════════════════════════════
# OBSERVATIONS
# ════════════════════════════════════════════════════════════════════════════

def load_observations(export_file: Path = EXPORT_FILE) -> pd.DataFrame:
    """Une ligne par image labellisée (avant téléchargement)."""
    obs = pd.read_csv(export_file, low_memory=False, encoding="utf-8-sig")
    obs["species_name"] = obs["Nom scientifique - observation"].str.strip().str.lower()
    identifiable = obs["espece identifiable ? - observation"].astype(str).str.lower()

    obs = obs[
        obs["species_name"].notna()
        & obs["images - observation"].notna()
        & ~identifiable.str.startswith("non")
    ]

    df = pd.DataFrame({
        "id_observation": obs["ID - observation"].astype(int),
        "id_n1": obs["ID - N1"].astype(int),
        "species_name": obs["species_name"],
        "url": obs["images - observation"].str.split("|"),
    }).explode("url")
    df["url"] = df["url"].str.strip()
    df = df[df["url"].str.startswith("http")].drop_duplicates(["id_observation", "url"])
    df["k"] = df.groupby("id_observation").cumcount()
    df["image_path"] = [str(IMAGES_DIR / f"{o}_{k}.jpg")
                        for o, k in zip(df["id_observation"], df["k"])]
    return df.drop(columns="k").reset_index(drop=True)


def attach_taxonomy(df: pd.DataFrame, taxref: pd.DataFrame) -> pd.DataFrame:
    tax = taxref[["species_name"] + SUPERVISED_LEVELS].drop_duplicates("species_name")
    df = df.merge(tax, on="species_name", how="left")
    missing = df["regne"].isna()
    if missing.any():
        print(f"  ⚠ {missing.sum()} images ({df.loc[missing, 'species_name'].nunique()} noms) "
              "absents de TAXREF → exclues")
    return df[~missing].reset_index(drop=True)


# ════════════════════════════════════════════════════════════════════════════
# TÉLÉCHARGEMENT
# ════════════════════════════════════════════════════════════════════════════

def _download(url: str, dest: Path, session: requests.Session) -> bool:
    if dest.exists():
        return True
    try:
        r = session.get(url, timeout=30)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
        img.save(dest, "JPEG", quality=95)
        return True
    except Exception:
        return False


def download_images(df: pd.DataFrame, workers: int = 8) -> pd.DataFrame:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    ok = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_download, r.url, Path(r.image_path), session): r.image_path
                   for r in df.itertuples()}
        for i, fut in enumerate(as_completed(futures), 1):
            ok[futures[fut]] = fut.result()
            if i % 500 == 0:
                print(f"  {i}/{len(df)} images traitées")

    df = df[df["image_path"].map(ok)].reset_index(drop=True)
    print(f"  {len(df)} images disponibles ({len(ok) - len(df)} échecs)")
    return df


def build_dataset(export_file: Path = EXPORT_FILE, workers: int = 8) -> pd.DataFrame:
    df = load_observations(export_file)
    print(f"  {len(df)} images labellisées, {df['species_name'].nunique()} noms scientifiques")
    df = attach_taxonomy(df, load_taxref())
    df = download_images(df, workers)
    df.to_parquet(DATASET_FILE, index=False)
    print(f"  Dataset → {DATASET_FILE} ({len(df)} images, "
          f"{df['species_name'].nunique()} espèces)")
    return df


def main():
    parser = argparse.ArgumentParser(description="Construction du dataset de classification")
    parser.add_argument("--export", type=Path, default=EXPORT_FILE)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    build_dataset(args.export, args.workers)


if __name__ == "__main__":
    main()
