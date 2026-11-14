"""FastAPI app: submit posts, async moderation, public listing/search, human review queue.

Demo notes: there is no authentication; the review queue is open. Do not expose it publicly.
"""

from __future__ import annotations

import io
import mimetypes
import tempfile
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from PIL import Image
from pydantic import BaseModel

from src.i18n import SUPPORTED, get_lang, t
from src.moderation.pipeline import ModerationPipeline
from src.moderation.rules import Verdict
from src.storage.base import BlobStore, PostStore
from src.storage.factory import from_env

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXT = {".mp4", ".mov", ".webm"}
MAX_BYTES = {"image": 25 * 1024 * 1024, "video": 100 * 1024 * 1024}
STATUS_OF = {Verdict.PUBLISH: "published", Verdict.REVIEW: "review", Verdict.REJECT: "rejected"}
TEMPLATES = Jinja2Templates(directory=Path(__file__).parent / "templates")


class ReviewBody(BaseModel):
    decision: str  # publish | reject


def create_app(pipeline_factory: Callable[[], ModerationPipeline] | None = None,
               data_dir: Path = Path("data/app"), db: PostStore | None = None,
               blobs: BlobStore | None = None) -> FastAPI:
    app = FastAPI(title="PawMatch Guard")
    if db is None or blobs is None:
        env_db, env_blobs = from_env(data_dir)
        db, blobs = db or env_db, blobs or env_blobs
    upload_key = lambda pid, ext: f"uploads/{pid}{ext}"  # noqa: E731
    public_key = lambda pid: f"public/{pid}.jpg"  # noqa: E731

    if pipeline_factory is None:
        def pipeline_factory() -> ModerationPipeline:
            from src.moderation.torch_models import load_pipeline

            return load_pipeline()

    get_pipeline = lru_cache(maxsize=1)(pipeline_factory)

    def lang_of(request: Request) -> str:
        return get_lang(request.query_params.get("lang") or request.cookies.get("lang"))

    def page(request: Request, name: str, **ctx):
        lang = lang_of(request)
        resp = TEMPLATES.TemplateResponse(request, name, {
            "t": lambda k, **kw: t(k, lang, **kw), "lang": lang, "langs": SUPPORTED, **ctx})
        resp.set_cookie("lang", lang)
        return resp

    def view(post: dict, lang: str) -> dict:
        return {
            "id": post["id"], "title": post["title"], "description": post["description"],
            "species": post["declared_species"], "detected_species": post["detected_species"],
            "media_type": post["media_type"], "status": post["status"],
            "reasons": post["reasons"],
            "reason_text": [t(f"reason.{r}", lang) for r in post["reasons"]],
        }

    # ---- moderation (runs in the background) -------------------------------------------
    def moderate(post_id: int) -> None:
        post = db.get(post_id)
        if post is None:
            return
        try:
            pipe = get_pipeline()
            raw = blobs.get(upload_key(post_id, post["ext"]))
            if post["media_type"] == "image":
                img = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
                res, shown = pipe.moderate_image(img, post["declared_species"])
            else:
                with tempfile.TemporaryDirectory() as tmp:
                    src = Path(tmp) / f"video{post['ext']}"
                    src.write_bytes(raw)
                    res = pipe.moderate_video(src, Path(tmp) / "frames",
                                              post["declared_species"])
                shown = res.thumbnail
            buf = io.BytesIO()
            Image.fromarray(shown).save(buf, "JPEG", quality=90)
            blobs.put(public_key(post_id), buf.getvalue())
            d = res.decision
            db.set_moderation(post_id, STATUS_OF[d.verdict], list(d.reasons),
                              d.extra.get("species"))
        except Exception:  # noqa: BLE001 - never leave a post stuck in "pending"
            db.set_moderation(post_id, "error", ["processing_failed"], None)

    # ---- API ---------------------------------------------------------------------------
    @app.post("/api/posts", status_code=202)
    async def create_post(background: BackgroundTasks, file: UploadFile = File(...),
                          title: str = Form(..., max_length=120),
                          description: str = Form("", max_length=2000),
                          species: str = Form(...)):
        if species not in ("cat", "dog"):
            raise HTTPException(422, "species must be 'cat' or 'dog'")
        ext = Path(file.filename or "").suffix.lower()
        kind = "image" if ext in IMAGE_EXT else "video" if ext in VIDEO_EXT else None
        if kind is None:
            raise HTTPException(415, f"unsupported file type: {ext or 'unknown'}")
        data = await file.read(MAX_BYTES[kind] + 1)
        if len(data) > MAX_BYTES[kind]:
            raise HTTPException(413, "file too large")
        post_id = db.create_post(title.strip(), description.strip(), species, kind, ext)
        blobs.put(upload_key(post_id, ext), data)  # key from id, never from user input
        background.add_task(moderate, post_id)
        return {"id": post_id, "status": "pending"}

    @app.get("/api/posts")
    def list_posts(request: Request, species: str | None = None, q: str | None = None):
        lang = lang_of(request)
        return [view(p, lang) for p in db.list_posts("published", species, q)]

    @app.get("/api/posts/{post_id}")
    def get_post(request: Request, post_id: int):
        post = db.get(post_id)
        if post is None:
            raise HTTPException(404)
        return view(post, lang_of(request))

    @app.get("/api/review")
    def review_queue(request: Request):
        lang = lang_of(request)
        return [view(p, lang) for p in db.list_posts("review")]

    @app.post("/api/review/{post_id}")
    def resolve(post_id: int, body: ReviewBody):
        if body.decision not in ("publish", "reject"):
            raise HTTPException(422, "decision must be 'publish' or 'reject'")
        new = "published" if body.decision == "publish" else "rejected"
        if not db.set_status_if(post_id, "review", new):
            raise HTTPException(409, "post is not awaiting review")
        return {"id": post_id, "status": new}

    @app.get("/media/{post_id}")
    def media(post_id: int):
        post = db.get(post_id)
        if post is None or post["status"] != "published":
            raise HTTPException(404)
        return Response(blobs.get(public_key(post_id)), media_type="image/jpeg")

    @app.get("/media/{post_id}/video")
    def video(post_id: int):
        post = db.get(post_id)
        if post is None or post["status"] != "published" or post["media_type"] != "video":
            raise HTTPException(404)
        mime = mimetypes.guess_type(f"x{post['ext']}")[0] or "application/octet-stream"
        return Response(blobs.get(upload_key(post_id, post["ext"])), media_type=mime)

    @app.get("/review/media/{post_id}")
    def review_media(post_id: int):
        post = db.get(post_id)
        if post is None or post["status"] != "review":
            raise HTTPException(404)
        return Response(blobs.get(public_key(post_id)), media_type="image/jpeg")

    # ---- pages -------------------------------------------------------------------------
    @app.get("/")
    def home(request: Request, species: str | None = None, q: str | None = None):
        lang = lang_of(request)
        posts = [view(p, lang) for p in db.list_posts("published", species, q)]
        return page(request, "index.html", posts=posts, species=species or "", q=q or "")

    @app.get("/submit")
    def submit_page(request: Request):
        return page(request, "submit.html")

    @app.get("/post/{post_id}")
    def post_page(request: Request, post_id: int):
        post = db.get(post_id)
        if post is None:
            raise HTTPException(404)
        return page(request, "post.html", post=view(post, lang_of(request)))

    @app.get("/review")
    def review_page(request: Request):
        lang = lang_of(request)
        return page(request, "review.html",
                    posts=[view(p, lang) for p in db.list_posts("review")])

    @app.post("/review/{post_id}")
    def review_form(post_id: int, decision: str = Form(...)):
        if decision in ("publish", "reject"):
            db.set_status_if(post_id, "review", "published" if decision == "publish" else
                             "rejected")
        return RedirectResponse("/review", status_code=303)

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)

    return app
