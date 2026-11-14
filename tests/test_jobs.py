import io

import numpy as np
import pytest
from PIL import Image

from src.api import jobs
from src.api.db import Database
from src.moderation.pipeline import ModerationPipeline
from src.moderation.rules import Detection
from src.moderation.weights import sync_weights
from src.storage.blobs import LocalBlobStore


class Det:
    def __init__(self):
        self.dets = [Detection("animal", 0.95, (0, 0, 9, 9))]

    def detect(self, image):
        return self.dets


class Clf:
    species = "cat"

    def predict(self, image):
        return self.species, 0.99


def png() -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.full((32, 32, 3), 100, np.uint8)).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def env(tmp_path):
    db, blobs = Database(tmp_path / "d.db"), LocalBlobStore(tmp_path / "b")
    det, clf = Det(), Clf()
    return db, blobs, det, ModerationPipeline(det, clf)


def add(db, blobs, species="cat"):
    pid = db.create_post("t", "d", species, "image", ".png")
    blobs.put(jobs.upload_key(pid, ".png"), png())
    return pid


def test_moderate_post_writes_result_and_public_image(env) -> None:
    db, blobs, _, pipe = env
    pid = add(db, blobs)
    assert jobs.moderate_post(db, blobs, pipe, pid) == "published"
    assert db.get(pid)["status"] == "published" and blobs.exists(jobs.public_key(pid))
    with pytest.raises(KeyError):
        jobs.moderate_post(db, blobs, pipe, 999)


def test_moderate_post_marks_error_when_media_is_missing(env) -> None:
    db, blobs, _, pipe = env
    pid = db.create_post("t", "d", "cat", "image", ".png")  # no blob stored
    assert jobs.moderate_post(db, blobs, pipe, pid) == "error"
    assert db.get(pid)["status"] == "error"


def test_reevaluate_only_tightens_and_respects_humans(env) -> None:
    db, blobs, det, pipe = env
    ok, human, rejected = add(db, blobs), add(db, blobs), add(db, blobs)
    for pid in (ok, human):
        jobs.moderate_post(db, blobs, pipe, pid)
    db.set_moderation(rejected, "rejected", ["no_animal"], None)
    # a person reviewed `human` and published it
    db.set_moderation(human, "review", ["species_mismatch"], "dog")
    assert db.set_status_if(human, "review", "published", human=True)

    det.dets = []  # the new model no longer sees an animal anywhere
    out = jobs.reevaluate_all(db, blobs, pipe)
    assert out["checked"] == 1 and out["skipped_human"] == 1
    assert [c["id"] for c in out["tightened"]] == [ok]
    assert db.get(ok)["status"] == "rejected"
    assert db.get(human)["status"] == "published"  # human decision untouched
    assert db.get(rejected)["status"] == "rejected"

    det.dets = [Detection("animal", 0.99, (0, 0, 9, 9))]  # model now accepts everything
    out2 = jobs.reevaluate_all(db, blobs, pipe)
    assert out2["tightened"] == [] and db.get(ok)["status"] == "rejected"  # never loosens


def test_sync_weights_writes_a_pipeline_config(tmp_path) -> None:
    blobs = LocalBlobStore(tmp_path / "s3")
    for part in ("classifier", "detector"):
        for name in ("config.yaml", "model.pt"):
            blobs.put(f"models/{part}/{name}", f"{part}-{name}".encode())
    cfg = sync_weights(blobs, "models/", tmp_path / "dest", tmp_path / "th.yaml")
    import yaml

    c = yaml.safe_load(cfg.read_text())
    assert (tmp_path / "dest" / "detector" / "model.pt").read_bytes() == b"detector-model.pt"
    assert c["classifier_run"].endswith("classifier") and c["thresholds"].endswith("th.yaml")


def test_moderate_post_is_idempotent_and_keeps_human_decisions(env) -> None:
    db, blobs, det, pipe = env
    pid = add(db, blobs)
    assert jobs.moderate_post(db, blobs, pipe, pid) == "published"
    det.dets = []  # a duplicate event arrives after the model changed
    assert jobs.moderate_post(db, blobs, pipe, pid) == "published"
    assert db.get(pid)["status"] == "published"
