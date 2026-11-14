"""Contract tests: every PostStore / BlobStore implementation must behave the same."""

import boto3
import pytest
from moto import mock_aws

from src.api.db import Database
from src.storage.blobs import LocalBlobStore, S3BlobStore
from src.storage.dynamo import DynamoPostStore, create_table


@pytest.fixture(params=["sqlite", "dynamodb"])
def posts(request, tmp_path):
    if request.param == "sqlite":
        yield Database(tmp_path / "t.db")
    else:
        with mock_aws():
            create_table(boto3.client("dynamodb", region_name="us-east-1"), "posts")
            yield DynamoPostStore("posts", boto3.resource("dynamodb", region_name="us-east-1"))


@pytest.fixture(params=["local", "s3"])
def blobs(request, tmp_path):
    if request.param == "local":
        yield LocalBlobStore(tmp_path / "b")
    else:
        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket="bkt")
            yield S3BlobStore("bkt", client=client)


def new(posts, title="Mimi", desc="calm cat", species="cat"):
    return posts.create_post(title, desc, species, "image", ".png")


def test_create_get_and_unique_increasing_ids(posts) -> None:
    a, b = new(posts), new(posts, "Rex", "dog", "dog")
    assert 0 < a < b
    p = posts.get(a)
    assert p["title"] == "Mimi" and p["status"] == "pending" and p["reasons"] == []
    assert p["detected_species"] is None and posts.get(9999) is None


def test_moderation_and_compare_and_set(posts) -> None:
    pid = new(posts)
    posts.set_moderation(pid, "review", ["species_mismatch"], "dog")
    p = posts.get(pid)
    assert (p["status"], p["reasons"], p["detected_species"]) == (
        "review", ["species_mismatch"], "dog")
    assert posts.set_status_if(pid, "published", "rejected") is False  # wrong expectation
    assert posts.set_status_if(pid, "review", "published") is True
    assert posts.set_status_if(pid, "review", "rejected") is False  # already resolved
    assert posts.get(pid)["status"] == "published"


def test_listing_filters_and_order(posts) -> None:
    ids = [new(posts, "Mimi", "calm"), new(posts, "Rex", "100% good_boy", "dog"),
           new(posts, "Tama", "shy cat")]
    for i, sp in zip(ids, ("cat", "dog", "cat"), strict=True):
        posts.set_moderation(i, "published", [], sp)
    posts.set_moderation(new(posts, "Hidden", "x"), "rejected", ["no_animal"], None)
    assert [p["id"] for p in posts.list_posts("published")] == ids[::-1]  # newest first
    assert [p["title"] for p in posts.list_posts("published", species="dog")] == ["Rex"]
    assert [p["title"] for p in posts.list_posts("published", q="MIM")] == ["Mimi"]
    assert [p["title"] for p in posts.list_posts("published", q="%")] == ["Rex"]  # literal
    assert len(posts.list_posts("published", limit=2)) == 2
    assert [p["title"] for p in posts.list_posts("rejected")] == ["Hidden"]


def test_blob_roundtrip(blobs) -> None:
    assert not blobs.exists("uploads/1.png")
    blobs.put("uploads/1.png", b"abc")
    assert blobs.exists("uploads/1.png") and blobs.get("uploads/1.png") == b"abc"


def test_local_blob_rejects_path_escape(tmp_path) -> None:
    store = LocalBlobStore(tmp_path / "b")
    with pytest.raises(ValueError):
        store.put("../evil", b"x")
    assert not (tmp_path / "evil").exists()


def test_human_flag_is_set_only_by_human_resolution(posts) -> None:
    a, b = new(posts), new(posts)
    for i in (a, b):
        posts.set_moderation(i, "review", ["species_mismatch"], "dog")
    assert posts.get(a)["human_reviewed"] is False
    posts.set_status_if(a, "review", "published", human=True)
    posts.set_status_if(b, "review", "rejected")  # automatic
    assert posts.get(a)["human_reviewed"] is True and posts.get(b)["human_reviewed"] is False
    posts.set_moderation(a, "published", [], "cat")  # re-moderation keeps the flag
    assert posts.get(a)["human_reviewed"] is True
