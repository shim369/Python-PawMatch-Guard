"""Pick storage from environment variables.

PAWMATCH_STORAGE=local (default): SQLite + local files under the data dir.
PAWMATCH_STORAGE=aws            : DynamoDB (PAWMATCH_TABLE) + S3 (PAWMATCH_BUCKET).
  PAWMATCH_AWS_ENDPOINT lets the same code talk to MinIO / local emulators.
"""

from __future__ import annotations

import os
from pathlib import Path

from .base import BlobStore, PostStore
from .blobs import LocalBlobStore, S3BlobStore


def from_env(data_dir: Path) -> tuple[PostStore, BlobStore]:
    if os.environ.get("PAWMATCH_STORAGE", "local") == "aws":
        from .dynamo import DynamoPostStore

        endpoint = os.environ.get("PAWMATCH_AWS_ENDPOINT")
        return (DynamoPostStore(os.environ["PAWMATCH_TABLE"], endpoint_url=endpoint),
                S3BlobStore(os.environ["PAWMATCH_BUCKET"], endpoint_url=endpoint))
    from src.api.db import Database

    return Database(data_dir / "app.db"), LocalBlobStore(data_dir)
