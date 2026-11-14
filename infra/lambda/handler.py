"""S3 ObjectCreated -> start one ECS Fargate task that moderates that post.

Only keys like ``uploads/<post_id>.<ext>`` are handled. The task exits when done, so nothing
runs (or is billed) while there are no uploads.
"""

import json
import os
import re
import urllib.parse

import boto3

KEY = re.compile(r"^uploads/(\d+)\.[A-Za-z0-9]+$")
_ecs = None


def _client():
    global _ecs
    if _ecs is None:
        _ecs = boto3.client("ecs")
    return _ecs


def handler(event, context=None, ecs=None):
    ecs = ecs or _client()
    started, skipped = [], []
    for rec in event.get("Records", []):
        key = urllib.parse.unquote_plus(rec["s3"]["object"]["key"])
        m = KEY.match(key)
        if not m:
            skipped.append(key)
            continue
        resp = ecs.run_task(
            cluster=os.environ["CLUSTER_ARN"],
            taskDefinition=os.environ["TASK_DEFINITION_ARN"],
            launchType="FARGATE",
            networkConfiguration={"awsvpcConfiguration": {
                "subnets": os.environ["SUBNET_IDS"].split(","),
                "securityGroups": [os.environ["SECURITY_GROUP_ID"]],
                "assignPublicIp": "ENABLED"}},
            overrides={"containerOverrides": [{
                "name": "worker", "command": ["moderate-post", "--id", m.group(1)]}]},
        )
        if resp.get("failures"):
            raise RuntimeError(f"run_task failed for {key}: {resp['failures']}")
        started.append(m.group(1))
    print(json.dumps({"started": started, "skipped": skipped}))
    return {"started": started, "skipped": skipped}
