import io

import numpy as np
import pytest
from PIL import Image

from src import aws_lambda
from src.api import jobs
from src.api.db import Database
from src.moderation.pipeline import ModerationPipeline
from src.moderation.rules import Detection
from src.storage.blobs import LocalBlobStore


class Det:
    dets = [Detection("animal", 0.95, (0, 0, 9, 9))]

    def detect(self, image):
        return list(self.dets)


class Clf:
    def predict(self, image):
        return "cat", 0.99


class FakeLambda:
    def __init__(self):
        self.calls = []

    def invoke(self, **kw):
        self.calls.append(kw)


@pytest.fixture
def deps(tmp_path):
    db, blobs = Database(tmp_path / "d.db"), LocalBlobStore(tmp_path / "b")
    return db, blobs, ModerationPipeline(Det(), Clf())


def add(db, blobs):
    pid = db.create_post("t", "d", "cat", "image", ".png")
    buf = io.BytesIO()
    Image.fromarray(np.full((32, 32, 3), 100, np.uint8)).save(buf, "PNG")
    blobs.put(jobs.upload_key(pid, ".png"), buf.getvalue())
    return pid


def s3_event(*keys):
    return {"Records": [{"s3": {"object": {"key": k}}} for k in keys]}


def test_s3_event_moderates_the_post_and_ignores_other_keys(deps) -> None:
    db, blobs, _ = deps
    pid = add(db, blobs)
    out = aws_lambda.handler(
        s3_event(f"uploads/{pid}.png", "public/1.jpg", "uploads/1.png; rm -rf", "uploads/../2.png"),
        deps=deps)
    assert out["moderated"] == {pid: "published"} and len(out["skipped"]) == 3
    assert db.get(pid)["status"] == "published"


def test_duplicate_s3_event_is_harmless(deps) -> None:
    db, blobs, _ = deps
    pid = add(db, blobs)
    ev = s3_event(f"uploads/{pid}.png")
    aws_lambda.handler(ev, deps=deps)
    assert aws_lambda.handler(ev, deps=deps)["moderated"] == {pid: "published"}


def test_unknown_post_raises_so_s3_retries(deps) -> None:
    with pytest.raises(KeyError):
        aws_lambda.handler(s3_event("uploads/999.png"), deps=deps)


def test_reevaluate_chains_itself_until_done(deps, monkeypatch) -> None:
    db, blobs, pipe = deps
    monkeypatch.setattr(aws_lambda, "CHUNK", 2)
    monkeypatch.setenv("AWS_LAMBDA_FUNCTION_NAME", "fn")
    ids = [add(db, blobs) for _ in range(3)]
    for i in ids:
        jobs.moderate_post(db, blobs, pipe, i)
    Det.dets = []
    try:
        client = FakeLambda()
        first = aws_lambda.handler({"action": "reevaluate"}, deps=deps, lambda_client=client)
        assert len(first["tightened"]) == 2 and len(client.calls) == 1
        assert client.calls[0]["InvocationType"] == "Event"
        assert client.calls[0]["FunctionName"] == "fn"
        import json

        nxt = json.loads(client.calls[0]["Payload"])
        assert nxt == {"action": "reevaluate", "after_id": ids[1]}
        last = aws_lambda.handler(nxt, deps=deps, lambda_client=client)
        assert len(last["tightened"]) == 1 and len(client.calls) == 1  # no further chain
    finally:
        Det.dets = [Detection("animal", 0.95, (0, 0, 9, 9))]
