"""
export_onnx.py — Conversion des modèles en ONNX
================================================
1. Encodeur visuel BioCLIP-2 (encode_image + normalisation L2)
       "pixel_values" (N, 3, 224, 224) → "features" (N, 768)
2. Têtes de classification (prototypes + régression logistique + fusion + agrégation)
       "features" (N, 768) → "p_species_name", "p_regne", …, "p_famille"
   (appelé par training/train.py via export_heads()).

Chaque export est vérifié immédiatement contre PyTorch (même entrée → même sortie) et
échoue si l'écart dépasse la tolérance : c'est la non-régression ONNX ↔ PyTorch.

Usage :
    uv run python -m ml.classification.training.export_onnx

Nécessite (export uniquement, pas au runtime) : torch, open_clip, onnx, onnxscript
"""

import argparse
import time
from pathlib import Path

import numpy as np

from ..classifier_bioclip import make_onnx_session
from ..config import BIOCLIP_MODEL_ID, ENCODER_ONNX_FILE, ONNX_OPSET


def export_encoder(output: Path = ENCODER_ONNX_FILE, model_id: str = BIOCLIP_MODEL_ID,
                   opset: int = ONNX_OPSET) -> Path:
    import open_clip
    import torch
    import torch.nn.functional as F

    class VisualEncoder(torch.nn.Module):
        def __init__(self, clip_model):
            super().__init__()
            self.clip_model = clip_model

        def forward(self, pixel_values):
            return F.normalize(self.clip_model.encode_image(pixel_values), dim=-1)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Chargement {model_id} (PyTorch)...")
    clip_model, _, _ = open_clip.create_model_and_transforms(model_id)
    encoder = VisualEncoder(clip_model.eval()).eval()

    print(f"  Export ONNX (opset {opset}) → {output}")
    t0 = time.time()
    torch.onnx.export(
        encoder, (torch.randn(2, 3, 224, 224),), str(output),
        input_names=["pixel_values"], output_names=["features"],
        dynamic_axes={"pixel_values": {0: "batch"}, "features": {0: "batch"}},
        opset_version=opset, dynamo=False, external_data=False,
    )
    print(f"  Export terminé en {time.time() - t0:.0f}s ({output.stat().st_size / 1e6:.0f} Mo)")

    check = torch.randn(8, 3, 224, 224)
    with torch.no_grad():
        ref = encoder(check).numpy()
    got = make_onnx_session(output).run(["features"], {"pixel_values": check.numpy()})[0]
    _report("encodeur", ref, got, cos_min=0.9999)
    return output


def export_heads(heads_module, output: Path, feature_dim: int, opset: int = ONNX_OPSET) -> Path:
    """Exporte le module PyTorch des têtes (components.FusionModel.to_torch())."""
    import torch

    heads_module = heads_module.eval()
    names = heads_module.output_names
    torch.onnx.export(
        heads_module, (torch.randn(3, feature_dim),), str(output),
        input_names=["features"], output_names=names,
        dynamic_axes={"features": {0: "batch"}, **{n: {0: "batch"} for n in names}},
        opset_version=opset, dynamo=False, external_data=False,
    )
    x = torch.nn.functional.normalize(torch.randn(64, feature_dim), dim=-1)
    with torch.no_grad():
        refs = heads_module(x)
    gots = make_onnx_session(output).run(names, {"features": x.numpy()})
    for name, ref, got in zip(names, refs, gots):
        _report(name, ref.numpy(), got, max_abs=1e-5)
    print(f"  Têtes ONNX → {output} ({output.stat().st_size / 1e6:.1f} Mo)")
    return output


def _report(name, ref, got, cos_min=None, max_abs=None):
    diff = float(np.abs(ref - got).max())
    msg = f"  Vérif. {name}: écart max={diff:.2e}"
    if cos_min is not None:
        cos = float((ref * got).sum(axis=1).min())
        msg += f" | cosinus min={cos:.7f}"
        if cos < cos_min:
            raise RuntimeError(f"{name}: ONNX diverge de PyTorch (cos={cos})")
    if max_abs is not None and diff > max_abs:
        raise RuntimeError(f"{name}: ONNX diverge de PyTorch (écart={diff})")
    print(msg)


def main():
    parser = argparse.ArgumentParser(description="Export de l'encodeur BioCLIP-2 en ONNX")
    parser.add_argument("--output", type=Path, default=ENCODER_ONNX_FILE)
    export_encoder(parser.parse_args().output)


if __name__ == "__main__":
    main()
