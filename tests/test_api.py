import io
import re

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import src.api.app as appmod
from src.api.app import create_app
from src.moderation.pipeline import ModerationPipeline
from src.moderation.rules import Detection


class FakeDetector:
    def __init__(self):
        self.dets = [Detection("animal", 0.95, (5, 5, 50, 50))]

    def detect(self, image):
        return self.dets


class FakeClassifier:
    def __init__(self):
        self.species = "cat"

    def predict(self, image):
        return self.species, 0.99


def png(color=120) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.full((64, 64, 3), color, np.uint8)).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def env(tmp_path):
    det, clf = FakeDetector(), FakeClassifier()
    app = create_app(lambda: ModerationPipeline(det, clf), tmp_path)
    return TestClient(app), det, clf


def post(client, species="cat", title="Mimi", desc="friendly cat", name="a.png", data=None):
    return client.post("/api/posts", data={"title": title, "description": desc,
                                           "species": species},
                       files={"file": (name, data or png(), "application/octet-stream")})


def test_publish_flow_and_listing(env) -> None:
    c, _, _ = env
    r = post(c)
    assert r.status_code == 202
    pid = r.json()["id"]
    j = c.get(f"/api/posts/{pid}").json()
    assert j["status"] == "published" and j["detected_species"] == "cat"
    assert [p["id"] for p in c.get("/api/posts").json()] == [pid]
    assert c.get(f"/media/{pid}").status_code == 200


def test_reject_is_hidden_everywhere(env) -> None:
    c, det, _ = env
    det.dets = []
    pid = post(c).json()["id"]
    j = c.get(f"/api/posts/{pid}").json()
    assert j["status"] == "rejected" and j["reasons"] == ["no_animal"]
    assert c.get("/api/posts").json() == []
    assert c.get(f"/media/{pid}").status_code == 404
    assert c.get(f"/review/media/{pid}").status_code == 404


def test_mismatch_goes_to_review_then_human_decides_once(env) -> None:
    c, _, clf = env
    clf.species = "dog"
    pid = post(c, species="cat").json()["id"]
    assert c.get(f"/api/posts/{pid}").json()["status"] == "review"
    assert c.get("/api/posts").json() == []  # not public yet
    assert [p["id"] for p in c.get("/api/review").json()] == [pid]
    assert c.get(f"/review/media/{pid}").status_code == 200
    assert c.post(f"/api/review/{pid}", json={"decision": "publish"}).status_code == 200
    assert [p["id"] for p in c.get("/api/posts").json()] == [pid]
    assert c.post(f"/api/review/{pid}", json={"decision": "reject"}).status_code == 409
    assert c.post(f"/api/review/{pid}", json={"decision": "maybe"}).status_code == 422


def test_search_and_species_filter(env) -> None:
    c, _, clf = env
    post(c, title="Mimi", desc="calm")
    clf.species = "dog"
    post(c, species="dog", title="Rex", desc="100% good_boy")
    assert [p["title"] for p in c.get("/api/posts", params={"species": "dog"}).json()] == ["Rex"]
    assert [p["title"] for p in c.get("/api/posts", params={"q": "mim"}).json()] == ["Mimi"]
    # SQL wildcards in user input are literals, not patterns
    assert c.get("/api/posts", params={"q": "%"}).json()[0]["title"] == "Rex"
    assert len(c.get("/api/posts", params={"q": "_"}).json()) == 1


def test_input_validation(env, monkeypatch) -> None:
    c, _, _ = env
    assert post(c, name="evil.exe").status_code == 415
    assert post(c, species="bird").status_code == 422
    monkeypatch.setitem(appmod.MAX_BYTES, "image", 10)
    assert post(c).status_code == 413
    assert c.get("/api/posts/999").status_code == 404


def test_path_traversal_filename_is_ignored(env, tmp_path) -> None:
    c, _, _ = env
    pid = post(c, name="../../evil.png").json()["id"]
    assert (tmp_path / "uploads" / f"{pid}.png").exists()
    assert not (tmp_path.parent / "evil.png").exists()


def test_pipeline_failure_marks_error_not_pending(tmp_path) -> None:
    def boom():
        raise RuntimeError("model missing")

    c = TestClient(create_app(boom, tmp_path))
    pid = post(c).json()["id"]
    j = c.get(f"/api/posts/{pid}").json()
    assert j["status"] == "error" and c.get("/api/posts").json() == []


def test_pages_render_in_both_languages_and_escape_html(env) -> None:
    c, _, _ = env
    post(c, title="<script>alert(1)</script>")
    en = c.get("/").text
    assert "Browse" in en and "<script>alert(1)</script>" not in en
    ja = c.get("/?lang=ja").text
    assert "さがす" in ja and 'lang="ja"' in ja
    assert "確認キュー" in c.get("/review?lang=ja").text
    assert c.get("/submit").status_code == 200


def test_full_flow_on_aws_backends(tmp_path) -> None:
    """The same app, wired to DynamoDB + S3 (moto), behaves identically."""
    import boto3
    from moto import mock_aws

    from src.storage.blobs import S3BlobStore
    from src.storage.dynamo import DynamoPostStore, create_table

    with mock_aws():
        create_table(boto3.client("dynamodb", region_name="us-east-1"), "posts")
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="bkt")
        app = create_app(
            lambda: ModerationPipeline(FakeDetector(), FakeClassifier()), tmp_path,
            db=DynamoPostStore("posts", boto3.resource("dynamodb", region_name="us-east-1")),
            blobs=S3BlobStore("bkt", client=s3))
        c = TestClient(app)
        pid = post(c).json()["id"]
        assert c.get(f"/api/posts/{pid}").json()["status"] == "published"
        assert c.get(f"/media/{pid}").status_code == 200
        assert s3.head_object(Bucket="bkt", Key=f"uploads/{pid}.png")
        assert [p["id"] for p in c.get("/api/posts").json()] == [pid]


def test_static_assets_are_versioned_and_served(env) -> None:
    c, _, _ = env
    html = c.get("/").text
    assert "/static/style.css?v=" in html and "/static/app.js?v=" in html
    css = c.get("/static/style.css")
    assert css.status_code == 200 and "--brand" in css.text
    # The page background must use separate properties: a `background` shorthand that puts a
    # colour before a gradient list is invalid CSS and silently drops the whole declaration.
    assert "background-color: var(--bg)" in css.text
    assert "background: var(--bg) var(--bg-grad)" not in css.text
    assert c.get("/static/app.js").status_code == 200


def test_pages_have_icons_theme_toggle_and_escape_user_text(env) -> None:
    c, _, _ = env
    pid = post(c, title="<b>x</b>").json()["id"]
    for url in ("/", "/submit", "/review", f"/post/{pid}"):
        text = c.get(url).text
        assert 'id="theme-toggle"' in text
        used = set(re.findall(r'<use href="#(i-[a-z-]+)"', text))
        defined = set(re.findall(r'<symbol id="(i-[a-z-]+)"', text))
        assert used and used <= defined, used - defined  # no icon without a definition
    assert "<b>x</b>" not in c.get("/").text and "&lt;b&gt;x&lt;/b&gt;" in c.get("/").text
