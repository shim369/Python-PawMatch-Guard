"""CPU latency (p50 / p95) and accuracy for PyTorch vs ONNX Runtime vs INT8, by resolution
and batch size. Everything runs on CPU so the numbers are comparable; they describe the
machine they were measured on, not AWS."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import yaml
from PIL import Image

from src.data.oxford_pet import DATASET_DIR
from src.data.split import read_split_csv
from src.data.transforms import eval_transform
from src.models.classifier import PetClassifier

from .onnx_export import export_classifier, quantize_dynamic_int8


def session(path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def latency(fn, x: np.ndarray, runs: int = 40, warmup: int = 5) -> dict:
    for _ in range(warmup):
        fn(x)
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(x)
        ts.append((time.perf_counter() - t0) * 1000)
    batch = x.shape[0]
    return {"p50_ms_per_image": float(np.percentile(ts, 50)) / batch,
            "p95_ms_per_image": float(np.percentile(ts, 95)) / batch}


def predict_all(fn, xs: np.ndarray, chunk: int = 32) -> tuple[np.ndarray, np.ndarray]:
    outs = [fn(xs[i:i + chunk]) for i in range(0, len(xs), chunk)]
    return (np.concatenate([o[0] for o in outs]), np.concatenate([o[1] for o in outs]))


def load_images(run_dir: Path, n: int, size: int):
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    root = Path(cfg["data_root"]) / DATASET_DIR / "images"
    entries = read_split_csv(Path(cfg["split_csv"]))["test"][:n]
    tf = eval_transform(size)
    x = np.stack([tf(Image.open(root / f"{e.name}.jpg").convert("RGB")).numpy() for e in entries])
    return x, np.array([e.species_id for e in entries]), np.array([e.breed_id for e in entries])


def run(run_dir: Path, work: Path, n_acc: int = 400) -> dict:
    fp32 = export_classifier(run_dir, work / "classifier_fp32.onnx")
    int8 = quantize_dynamic_int8(fp32, work / "classifier_int8.onnx")
    model = PetClassifier(pretrained=False)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu"))
    model.eval()

    def torch_fn(x):
        with torch.no_grad():
            return [o.numpy() for o in model(torch.from_numpy(x))]

    s32, s8 = session(fp32), session(int8)
    ort_fn = lambda s: (lambda x: s.run(None, {"image": x}))  # noqa: E731
    variants = {"pytorch_fp32": torch_fn, "onnx_fp32": ort_fn(s32), "onnx_int8_dynamic": ort_fn(s8)}

    rows = []
    for size in (224, 160):
        xs, ys, yb = load_images(run_dir, n_acc, size)
        ref = predict_all(torch_fn, xs)[1].argmax(1)  # PyTorch fp32 at this resolution
        for name, fn in variants.items():
            sp, br = predict_all(fn, xs)
            acc = {"species_acc": float((sp.argmax(1) == ys).mean()),
                   "breed_acc": float((br.argmax(1) == yb).mean()),
                   "breed_agree_with_pytorch_fp32": float((br.argmax(1) == ref).mean())}
            for batch in (1, 8):
                lat = latency(fn, xs[:batch])
                rows.append({"variant": name, "size": size, "batch": batch, **lat, **acc})
                print(rows[-1], flush=True)
    return {
        "machine": f"{platform.processor() or platform.machine()} / {platform.platform()}",
        "torch": torch.__version__, "onnxruntime": ort.__version__,
        "cpu_threads_torch": torch.get_num_threads(), "n_accuracy_images": n_acc,
        "model_size_mb": {"fp32": fp32.stat().st_size / 1e6, "int8": int8.stat().st_size / 1e6},
        "rows": rows,
    }


def main(run_dir: str, out: str, work: str) -> None:
    res = run(Path(run_dir), Path(work))
    Path(out).write_text(json.dumps(res, indent=2))
