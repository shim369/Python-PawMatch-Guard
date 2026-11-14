import numpy as np
import onnxruntime as ort
import torch

from src.models.classifier import PetClassifier
from src.optimize.onnx_export import export_classifier, quantize_dynamic_int8


def test_onnx_matches_pytorch_and_supports_dynamic_shapes(tmp_path) -> None:
    torch.manual_seed(0)
    model = PetClassifier(pretrained=False).eval()
    run = tmp_path / "run"
    run.mkdir()
    torch.save(model.state_dict(), run / "model.pt")

    onnx_path = export_classifier(run, tmp_path / "m.onnx")
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    for batch, size in ((2, 224), (1, 160)):
        x = torch.randn(batch, 3, size, size)
        with torch.no_grad():
            ref = [o.numpy() for o in model(x)]
        out = sess.run(None, {"image": x.numpy()})
        for a, b in zip(ref, out, strict=True):
            assert a.shape == b.shape
            assert np.abs(a - b).max() < 1e-3

    q = quantize_dynamic_int8(onnx_path, tmp_path / "q.onnx")
    assert q.stat().st_size < onnx_path.stat().st_size / 2
    qs = ort.InferenceSession(str(q), providers=["CPUExecutionProvider"])
    x = np.random.randn(1, 3, 224, 224).astype(np.float32)
    assert qs.run(None, {"image": x})[1].shape == (1, 37)
