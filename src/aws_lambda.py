"""AWS Lambda entry point (container image). One function, two jobs:

* S3 ObjectCreated for ``uploads/<post_id>.<ext>`` -> moderate that post.
* ``{"action": "reevaluate"}`` -> re-moderate live posts after a model update, 20 posts per
  invocation, then invoke itself asynchronously with the next cursor until done (Lambda is
  limited to 15 minutes per invocation).

Runs entirely inside the AWS Lambda / DynamoDB / S3 free tiers at portfolio traffic; see
README "Cost" for the numbers and the one item that is not free.
"""

from __future__ import annotations

import json
import os
import re
import urllib.parse

from src.api import jobs

KEY = re.compile(r"^uploads/(\d+)\.[A-Za-z0-9]+$")
CHUNK = 20
_deps = None


def _default_deps():
    """(db, blobs, pipeline), created once per warm container."""
    global _deps
    if _deps is None:
        from pathlib import Path

        from src.moderation.torch_models import load_pipeline_from_env
        from src.storage.factory import from_env

        db, blobs = from_env(Path("/tmp/app"))
        _deps = (db, blobs, load_pipeline_from_env())
    return _deps


def _invoke_self(payload: dict, client=None) -> None:
    import boto3

    (client or boto3.client("lambda")).invoke(
        FunctionName=os.environ["AWS_LAMBDA_FUNCTION_NAME"],
        InvocationType="Event", Payload=json.dumps(payload).encode())


def handler(event, context=None, deps=None, lambda_client=None):
    db, blobs, pipe = deps or _default_deps()

    if event.get("action") == "reevaluate":
        out = jobs.reevaluate_all(db, blobs, pipe, after_id=int(event.get("after_id", 0)),
                                  max_posts=CHUNK)
        if out["next_after_id"] is not None:
            _invoke_self({"action": "reevaluate", "after_id": out["next_after_id"]},
                         lambda_client)
        print(json.dumps(out))
        return out

    done, skipped = {}, []
    for rec in event.get("Records", []):
        key = urllib.parse.unquote_plus(rec["s3"]["object"]["key"])
        m = KEY.match(key)
        if not m:
            skipped.append(key)
            continue
        post_id = int(m.group(1))
        # KeyError (row not visible yet) propagates so the async invocation is retried.
        done[post_id] = jobs.moderate_post(db, blobs, pipe, post_id)
    print(json.dumps({"moderated": done, "skipped": skipped}))
    return {"moderated": done, "skipped": skipped}
