"""Moderation jobs shared by the web app (background task), the ECS worker (one post per
task) and the Batch re-evaluation job."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from src.moderation.pipeline import ModerationPipeline
from src.moderation.rules import Verdict
from src.storage.base import BlobStore, PostStore

STATUS_OF = {Verdict.PUBLISH: "published", Verdict.REVIEW: "review", Verdict.REJECT: "rejected"}
SEVERITY = {"published": 0, "review": 1, "rejected": 2}


def upload_key(post_id: int, ext: str) -> str:
    return f"uploads/{post_id}{ext}"


def public_key(post_id: int) -> str:
    return f"public/{post_id}.jpg"


def run_moderation(post: dict, blobs: BlobStore, pipe: ModerationPipeline):
    """Returns (decision, image to show). Does not write to the post store."""
    raw = blobs.get(upload_key(post["id"], post["ext"]))
    if post["media_type"] == "image":
        img = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
        res, shown = pipe.moderate_image(img, post["declared_species"])
        return res.decision, shown
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / f"video{post['ext']}"
        src.write_bytes(raw)
        res = pipe.moderate_video(src, Path(tmp) / "frames", post["declared_species"])
    return res.decision, res.thumbnail


def moderate_post(db: PostStore, blobs: BlobStore, pipe: ModerationPipeline,
                  post_id: int) -> str:
    """Moderate one *pending* post and store the result. Never leaves a post 'pending'.

    Idempotent: S3 can deliver an event twice, so a post that already has a result (or a
    human decision) is returned unchanged instead of being moderated again.
    """
    post = db.get(post_id)
    if post is None:
        raise KeyError(f"post {post_id} not found")
    if post["status"] != "pending":
        return post["status"]
    try:
        decision, shown = run_moderation(post, blobs, pipe)
        buf = io.BytesIO()
        Image.fromarray(shown).save(buf, "JPEG", quality=90)
        blobs.put(public_key(post_id), buf.getvalue())
        status = STATUS_OF[decision.verdict]
        db.set_moderation(post_id, status, list(decision.reasons),
                          decision.extra.get("species"))
    except Exception:  # noqa: BLE001
        db.set_moderation(post_id, "error", ["processing_failed"], None)
        status = "error"
    return status


def reevaluate_all(db: PostStore, blobs: BlobStore, pipe: ModerationPipeline,
                   limit: int = 100_000) -> dict:
    """Re-run moderation on live posts after a model update.

    Safety rules: only *tightens* (published -> review/rejected, review -> rejected); never
    loosens, and never touches posts a person already decided on (``human_reviewed``).
    Rejected posts are left alone. Changes are returned for auditing.
    """
    summary = {"checked": 0, "skipped_human": 0, "unchanged": 0, "tightened": [], "errors": []}
    for status in ("published", "review"):
        for post in db.list_posts(status, limit=limit):
            if post["human_reviewed"]:
                summary["skipped_human"] += 1
                continue
            summary["checked"] += 1
            try:
                decision, shown = run_moderation(post, blobs, pipe)
            except Exception as e:  # noqa: BLE001
                summary["errors"].append({"id": post["id"], "error": type(e).__name__})
                continue
            new = STATUS_OF[decision.verdict]
            if SEVERITY[new] > SEVERITY[post["status"]]:
                buf = io.BytesIO()
                Image.fromarray(shown).save(buf, "JPEG", quality=90)
                blobs.put(public_key(post["id"]), buf.getvalue())
                db.set_moderation(post["id"], new, list(decision.reasons),
                                  decision.extra.get("species"))
                summary["tightened"].append({"id": post["id"], "from": post["status"],
                                             "to": new, "reasons": list(decision.reasons)})
            else:
                summary["unchanged"] += 1
    return summary
