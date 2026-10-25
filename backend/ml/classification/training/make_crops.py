"""
make_crops.py — Crops YOLO du dataset, à l'identique de la production
======================================================================
En production, la classification reçoit des CROPS : YOLO (ml/crop_inference,
config.yaml : conf, iou, imgsz, max_det=1) détecte l'objet principal, la boîte la
plus confiante est découpée, les images sans détection ne sont pas classifiées.

Ce script applique exactement ce traitement (fonctions de ml/crop_inference,
non modifiées) aux images du dataset, pour évaluer — et éventuellement entraîner —
la classification dans les conditions de production.

Sortie : data/dataset_biolit/crops/<id>.jpg + crops.parquet
    image_path (crop), source_path, split, yolo_class, yolo_conf + colonnes du dataset

Usage :
    uv run python -m ml.classification.training.make_crops --split test
    uv run python -m ml.classification.training.make_crops --split train val
"""

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image

from ..config import DATA_DIR
from .build_dataset import DATASET_FILE

CROPS_DIR = DATA_DIR / "crops"
CROPS_FILE = DATA_DIR / "crops.parquet"
YOLO_CONFIG = Path(__file__).resolve().parents[2] / "crop_inference" / "config.yaml"
BATCH = 32


def make_crops(splits: list[str]) -> pd.DataFrame:
    from ml.crop_inference.predict import load_config, load_model

    cfg = load_config(str(YOLO_CONFIG))
    infer = cfg["inference"]
    model = load_model(cfg)

    df = pd.read_parquet(DATASET_FILE).merge(pd.read_parquet(DATA_DIR / "split.parquet"),
                                             on="image_path")
    df = df[df["split"].isin(splits)].reset_index(drop=True)
    done = pd.read_parquet(CROPS_FILE) if CROPS_FILE.exists() else pd.DataFrame()
    if len(done):
        df = df[~df["image_path"].isin(done["source_path"])].reset_index(drop=True)
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  {len(df)} images à traiter (splits {splits})")

    rows = []
    for i in range(0, len(df), BATCH):
        chunk = df.iloc[i:i + BATCH]
        results = model.predict(source=chunk["image_path"].tolist(), conf=infer["conf"],
                                iou=infer["iou"], imgsz=infer["imgsz"], device=infer["device"],
                                save=False, max_det=1, verbose=False)
        for (_, row), r in zip(chunk.iterrows(), results):
            if len(r.boxes) == 0:
                continue                                   # pas de crop → non classifié en prod
            box = r.boxes[r.boxes.conf.argmax()]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            crop = Image.open(row["image_path"]).convert("RGB").crop((x1, y1, x2, y2))
            out = CROPS_DIR / Path(row["image_path"]).name
            crop.save(out, quality=95)
            rows.append({**row.to_dict(), "source_path": row["image_path"], "image_path": str(out),
                         "yolo_class": r.names[int(box.cls)], "yolo_conf": float(box.conf)})
        print(f"  {min(i + BATCH, len(df))}/{len(df)} images, {len(rows)} crops")

    crops = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    crops.to_parquet(CROPS_FILE, index=False)
    print(f"  {len(crops)} crops → {CROPS_FILE}")
    return crops


def main():
    parser = argparse.ArgumentParser(description="Crops YOLO du dataset (comme en production)")
    parser.add_argument("--split", nargs="+", default=["test"])
    args = parser.parse_args()
    make_crops(args.split)


if __name__ == "__main__":
    main()
