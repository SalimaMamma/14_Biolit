"""
extract_features.py — Features BioCLIP-2 du dataset (avec cache)
=================================================================
Extrait les features de toutes les images de dataset.parquet (ou des crops YOLO de
crops.parquet) avec l'encodeur ONNX, le même qu'en production.

Sortie : data/dataset_biolit/features_<nom>.npz  { image_path, features (N, 768) }

Usage :
    uv run python -m ml.classification.training.extract_features
    uv run python -m ml.classification.training.extract_features --dataset crops --name crops_fp32
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ..classifier_bioclip import BioCLIPOnnxExtractor
from ..config import DATA_DIR, ENCODER_ONNX_FILE
from .build_dataset import DATASET_FILE

CHUNK = 256


def features_file(name: str) -> Path:
    return DATA_DIR / f"features_{name}.npz"


def load_features(name: str = "fp32") -> dict[str, np.ndarray]:
    """{image_path → feature (768,)}"""
    path = features_file(name)
    if not path.exists():
        return {}
    d = np.load(path, allow_pickle=False)
    return dict(zip(d["image_path"].tolist(), d["features"]))


def extract_dataset(paths: list[str], extractor, name: str) -> dict[str, np.ndarray]:
    cache = load_features(name)
    todo = [p for p in paths if p not in cache]
    print(f"  {len(paths) - len(todo)} features en cache, {len(todo)} à extraire")
    t0 = time.time()
    for i in range(0, len(todo), CHUNK):
        chunk = todo[i:i + CHUNK]
        images = [Image.open(p).convert("RGB") for p in chunk]
        for p, f in zip(chunk, extractor.extract_batch(images)):
            cache[p] = f
        done = i + len(chunk)
        rate = done / (time.time() - t0)
        print(f"  {done}/{len(todo)} ({rate:.1f} img/s, reste ~{(len(todo) - done) / rate / 60:.0f} min)")
        _save(cache, name)
    return cache


def _save(cache: dict, name: str):
    keys = sorted(cache)
    np.savez(features_file(name), image_path=np.array(keys),
             features=np.stack([cache[k] for k in keys]).astype(np.float32))


def main():
    parser = argparse.ArgumentParser(description="Extraction features BioCLIP-2 (ONNX)")
    parser.add_argument("--dataset", choices=["images", "crops"], default="images",
                        help="crops : crops YOLO produits par make_crops.py")
    parser.add_argument("--name", type=str, default="fp32", help="nom du cache features_<nom>.npz")
    args = parser.parse_args()

    if args.dataset == "crops":
        from .make_crops import CROPS_FILE
        paths = pd.read_parquet(CROPS_FILE)["image_path"].tolist()
    else:
        paths = pd.read_parquet(DATASET_FILE)["image_path"].tolist()
    extract_dataset(paths, BioCLIPOnnxExtractor(ENCODER_ONNX_FILE), args.name)


if __name__ == "__main__":
    main()
