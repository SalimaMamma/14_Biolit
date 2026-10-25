"""
config.py — Configuration centrale de la classification
"""

import os
from pathlib import Path

# ── Chemins ───────────────────────────────────────────────────────────────────
ROOT_DIR      = Path(__file__).parent
DATA_DIR      = ROOT_DIR / "data/dataset_biolit"
IMAGES_DIR    = DATA_DIR / "images"
RESULTS_DIR   = ROOT_DIR / "results"

# ── Niveaux taxonomiques ───────────────────────────────────────────────────────
SUPERVISED_LEVELS = ["regne", "phylum", "classe", "ordre", "famille"]          # large → fin
DECISION_LEVELS   = ["species_name", "famille", "ordre", "classe", "phylum", "regne"]  # fin → large

# ── BioCLIP-2 (encodeur visuel, gelé) ─────────────────────────────────────────
BIOCLIP_MODEL_ID = "hf-hub:imageomics/bioclip-2"
IMG_SIZE         = 224
FEATURE_DIM      = 768    # dimension de sortie de BioCLIP-2 (ViT-L/14)
BATCH_SIZE       = 32     # batch d'extraction

# ── ONNX Runtime ───────────────────────────────────────────────────────────────
ONNX_OPSET   = 17
ONNX_THREADS = int(os.getenv("CLASSIF_ONNX_THREADS", "0"))   # 0 = auto

# ── Entraînement (training/) ───────────────────────────────────────────────────
SEED              = 42
DUPLICATE_COS     = 0.95  # photos val/test exclues si une quasi-jumelle est dans un split précédent
WHITENING_DIM     = 256   # 768d → 256d (PCA whitening) pour les prototypes
MIN_PROTO_IMAGES  = 3     # nb min. de photos d'entraînement pour qu'une espèce soit apprise
PROTO_ALPHA       = 0.6   # prototype = α × moyenne visuelle + (1 − α) × texte BioCLIP-2
LR_C_GRID         = [0.003, 0.01, 0.03, 0.1]   # régularisation de la régression logistique (CV)
FUSION_WEIGHTS    = [0.25, 0.5, 0.75]          # poids de la régression logistique dans la fusion
N_COMMON_SPECIES  = 50    # évaluation : « communes » = 50 espèces les plus photographiées

# ── Fichiers du modèle (produits par training/, publiés sur Hugging Face) ─────
ENCODER_ONNX_FILE = RESULTS_DIR / "bioclip2_visual.onnx"     # image → features 768d
HEADS_ONNX_FILE   = RESULTS_DIR / "classifier_heads.onnx"    # features → probabilités par niveau
META_FILE         = RESULTS_DIR / "classifier_meta.json"     # classes, taxonomie, seuils

# "huggingface" (défaut, production) ou "local" (fichiers de RESULTS_DIR)
MODEL_SOURCE = os.getenv("CLASSIF_MODEL_SOURCE", "huggingface").lower()
HF_REPO_ID   = "DataForGood/biolit-species-identification"
HF_SUBFOLDER = "classification"     # sous-dossier du dépôt contenant les fichiers du modèle
