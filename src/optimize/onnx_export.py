"""PyTorch -> ONNX export of the classifier, plus ONNX Runtime dynamic quantization."""

from __future__ import annotations

from pathlib import Path

import torch

from src.models.classifier import PetClassifier


def export_classifier(run_dir: Path, out: Path, size: int = 224) -> Path:
    """Export with dynamic batch/height/width (ResNet global pooling handles any size)."""
    model = PetClassifier(pretrained=False)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu"))
    model.eval()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, torch.randn(1, 3, size, size), str(out), input_names=["image"],
        output_names=["species_logits", "breed_logits"], opset_version=17, dynamo=False,
        dynamic_axes={"image": {0: "batch", 2: "h", 3: "w"},
                      "species_logits": {0: "batch"}, "breed_logits": {0: "batch"}},
    )
    return out


def quantize_dynamic_int8(src: Path, dst: Path) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
    return dst
