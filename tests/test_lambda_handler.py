import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "handler", Path(__file__).parents[1] / "infra" / "lambda" / "handler.py")
handler_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handler_mod)


class FakeEcs:
    def __init__(self, failures=None):
        self.calls, self.failures = [], failures or []

    def run_task(self, **kw):
        self.calls.append(kw)
        return {"failures": self.failures}


def ev(*keys):
    return {"Records": [{"s3": {"object": {"key": k}}} for k in keys]}


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("CLUSTER_ARN", "arn:cluster")
    monkeypatch.setenv("TASK_DEFINITION_ARN", "arn:task")
    monkeypatch.setenv("SUBNET_IDS", "subnet-1,subnet-2")
    monkeypatch.setenv("SECURITY_GROUP_ID", "sg-1")


def test_starts_one_task_per_upload_with_the_post_id() -> None:
    ecs = FakeEcs()
    out = handler_mod.handler(ev("uploads/12.png", "uploads/7.mp4"), ecs=ecs)
    assert out["started"] == ["12", "7"]
    c = ecs.calls[0]
    assert c["overrides"]["containerOverrides"][0]["command"] == ["moderate-post", "--id", "12"]
    assert c["networkConfiguration"]["awsvpcConfiguration"]["subnets"] == ["subnet-1", "subnet-2"]


def test_ignores_other_keys_and_odd_names() -> None:
    ecs = FakeEcs()
    keys = ["public/1.jpg", "models/classifier/model.pt", "uploads/x.png",
            "uploads/1.png; rm -rf", "uploads/../2.png"]
    out = handler_mod.handler(ev(*keys), ecs=ecs)
    assert out["started"] == [] and ecs.calls == [] and len(out["skipped"]) == len(keys)


def test_failed_run_task_raises_so_lambda_retries() -> None:
    with pytest.raises(RuntimeError):
        handler_mod.handler(ev("uploads/1.png"), ecs=FakeEcs([{"reason": "RESOURCE:CPU"}]))
