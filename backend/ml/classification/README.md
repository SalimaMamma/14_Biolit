# Classification taxonomique

Module appelé par `pipelines/run.py` pour classifier les crops produits par
`ml/crop_inference` : `flow_ml_classification(crops_images, df_crops)`.

## Méthode (v3 « fusion hiérarchique »)

```text
crop ──BioCLIP-2 gelé (ONNX)──▶ vecteur 768d
     ──têtes (ONNX)──▶ P(espèce) = w × régression logistique + (1 − w) × prototypes
                       P(famille), P(ordre)… = somme des P(espèce) du taxon
     ──décision──▶ niveau le plus fin dont la probabilité atteint son seuil
```

- **Prototypes** : un vecteur par espèce (≥ 3 photos) = moyenne visuelle + texte BioCLIP-2.
- **Régression logistique** : multinomiale sur les vecteurs 768d standardisés.
- **Décision** : espèce si P ≥ seuil espèce, sinon famille si P ≥ seuil famille, etc.
  Les colonnes renvoyées sont cohérentes : niveaux supérieurs issus de TAXREF, niveaux plus
  fins que `best_level` à `None`.

Seuils, poids de fusion et températures sont choisis à l'entraînement sur un jeu de
validation, sous contrainte de ne jamais faire moins bien que la méthode v2 ; ils sont stockés
dans `classifier_meta.json`. L'inférence n'utilise que **onnxruntime + numpy + PIL**.

## Fichiers

```text
classifier_bioclip.py      # prétraitement numpy + extracteurs BioCLIP-2 (ONNX / PyTorch)
classifier_infer.py        # chargement du modèle, têtes ONNX, décision hiérarchique
config.py                  # chemins, fichiers du modèle, hyper-paramètres
pipeline_classification.py # orchestration appelée par le pipeline principal
training/                  # reconstruction du dataset, export ONNX, entraînement, évaluation
tests/                     # tests unitaires et de non-régression
```

`training/` :

| Script | Rôle |
|---|---|
| `build_dataset.py` | export Biolit → images téléchargées + taxonomie TAXREF |
| `export_onnx.py` | export de l'encodeur BioCLIP-2 en ONNX (+ têtes, appelé par `train.py`) |
| `extract_features.py` | vecteurs BioCLIP-2 du dataset ou des crops (cache) |
| `data.py` | découpage train/val/test par sortie + purge des quasi-doublons |
| `components.py` | prototypes, régression logistique, fusion, agrégation, températures |
| `evaluation.py` | décision vectorisée, métriques, contraintes de non-régression, bootstrap |
| `baseline_v2.py`, `mlp.py` | méthode v2 ré-entraînée : référence de non-régression |
| `train.py` | entraînement complet, export, évaluation sur test |
| `make_crops.py`, `eval_crops.py` | évaluation sur crops YOLO (conditions de production) |

## Fichiers du modèle (Hugging Face `DataForGood/biolit-species-identification`, dossier `classification/`)

| Fichier | Contenu |
|---|---|
| `bioclip2_visual.onnx` | encodeur visuel BioCLIP-2 FP32 (image → 768d), 1,2 Go |
| `classifier_heads.onnx` | prototypes + régression logistique + fusion + agrégation par niveau |
| `classifier_meta.json` | classes par niveau, taxonomie, seuils, paramètres de fusion |

`CLASSIF_MODEL_SOURCE=local` charge ces fichiers depuis `results/` au lieu de Hugging Face.

## Ré-entraînement

```bash
cd backend
uv pip install onnx onnxscript                                   # export ONNX uniquement
# export Biolit dans ml/classification/data/dataset_biolit/export_biolit.csv
uv run python -m ml.classification.training.build_dataset        # images + TAXREF
uv run python -m ml.classification.training.export_onnx          # encodeur (une seule fois)
uv run python -m ml.classification.training.extract_features     # ~3 h CPU, avec cache
uv run python -m ml.classification.training.train                # ~1 h
uv run python -m ml.classification.training.make_crops --split test
uv run python -m ml.classification.training.extract_features --dataset crops --name crops_fp32
uv run python -m ml.classification.training.eval_crops
RUN_ONNX_REGRESSION=1 uv run pytest ml/classification/tests -vv
```

Supprimer `data/dataset_biolit/split.parquet` pour refaire le découpage
(par exemple après ajout de nouvelles photos).

## Tests

```bash
uv run pytest ml/classification/tests                           # rapides (CI)
RUN_ONNX_REGRESSION=1 uv run pytest ml/classification/tests -vv  # + intégration et performances
```
