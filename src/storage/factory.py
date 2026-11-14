"""Pick storage from environment variables.

PAWMATCH_STORAGE=local (default): SQLite + local files under the data dir.
PAWMATCH_STORAGE=aws            : DynamoDB (PAWMATCH_TABLE) + S3 (PAWMATCH_BUCKET).
  PAWMATCH_S3_ENDPOINT / PAWMATCH_DYNAMODB_ENDPOINT point the same code at MinIO and
  DynamoDB Local (leave unset for real AWS).
"""

from __future__ import annotations

import os
from pathlib import Path

from .base import BlobStore, PostStore
from .blobs import LocalBlobStore, S3BlobStore


def from_env(data_dir: Path) -> tuple[PostStore, BlobStore]:
    if os.environ.get("PAWMATCH_STORAGE", "local") == "aws":
        from .dynamo import DynamoPostStore

        return (DynamoPostStore(os.environ["PAWMATCH_TABLE"],
                                endpoint_url=os.environ.get("PAWMATCH_DYNAMODB_ENDPOINT")),
                S3BlobStore(os.environ["PAWMATCH_BUCKET"],
                            endpoint_url=os.environ.get("PAWMATCH_S3_ENDPOINT")))
    from src.api.db import Database

    return Database(data_dir / "app.db"), LocalBlobStore(data_dir)
