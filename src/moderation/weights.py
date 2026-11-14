"""Fetch trained weights from object storage (S3 / MinIO) for the worker containers.

Layout under ``<prefix>/``: ``classifier/{config.yaml,model.pt}`` and
``detector/{config.yaml,model.pt}``. Weights are not in Git or in the Docker image, so the
model can be updated by uploading new files and re-running the Batch re-evaluation job.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from src.storage.base import BlobStore

FILES = ("config.yaml", "model.pt")


def sync_weights(blobs: BlobStore, prefix: str, dest: Path, thresholds: Path) -> Path:
    """Download weights into ``dest`` and write a pipeline config pointing at them."""
    for part in ("classifier", "detector"):
        (dest / part).mkdir(parents=True, exist_ok=True)
        for name in FILES:
            (dest / part / name).write_bytes(blobs.get(f"{prefix.strip('/')}/{part}/{name}"))
    cfg = {"classifier_run": str(dest / "classifier"), "detector_run": str(dest / "detector"),
           "thresholds": str(thresholds)}
    out = dest / "pipeline.yaml"
    out.write_text(yaml.safe_dump(cfg))
    return out
