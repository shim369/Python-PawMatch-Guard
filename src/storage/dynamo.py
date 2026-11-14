"""PostStore on DynamoDB.

Table: partition key ``id`` (N). Item id 0 is an atomic counter. A GSI ``status-id-index``
(partition ``status``, sort ``id``) lists posts by status, newest first. Species / text
filters are applied in Python after the query, which is fine at portfolio scale; at larger
scale add a search service instead of scanning.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

COUNTER_ID = 0
GSI = "status-id-index"


def create_table(client, name: str) -> None:
    """Same schema as infra/terraform (used by tests and local development)."""
    client.create_table(
        TableName=name, BillingMode="PAY_PER_REQUEST",
        AttributeDefinitions=[{"AttributeName": "id", "AttributeType": "N"},
                              {"AttributeName": "status", "AttributeType": "S"}],
        KeySchema=[{"AttributeName": "id", "KeyType": "HASH"}],
        GlobalSecondaryIndexes=[{
            "IndexName": GSI, "Projection": {"ProjectionType": "ALL"},
            "KeySchema": [{"AttributeName": "status", "KeyType": "HASH"},
                          {"AttributeName": "id", "KeyType": "RANGE"}]}],
    )


class DynamoPostStore:
    def __init__(self, table_name: str, resource=None, endpoint_url: str | None = None) -> None:
        import boto3

        res = resource or boto3.resource("dynamodb", endpoint_url=endpoint_url)
        self.table = res.Table(table_name)

    def create_post(self, title: str, description: str, species: str, media_type: str,
                    ext: str) -> int:
        n = self.table.update_item(
            Key={"id": COUNTER_ID}, UpdateExpression="ADD n :one",
            ExpressionAttributeValues={":one": 1}, ReturnValues="UPDATED_NEW")["Attributes"]["n"]
        post_id = int(n)
        self.table.put_item(Item={
            "id": post_id, "title": title, "description": description,
            "declared_species": species, "media_type": media_type, "ext": ext,
            "status": "pending", "reasons": "[]", "created_at": datetime.now(UTC).isoformat()},
            ConditionExpression="attribute_not_exists(id)")
        return post_id

    def get(self, post_id: int) -> dict | None:
        item = self.table.get_item(Key={"id": post_id}).get("Item")
        return self._row(item) if item and post_id != COUNTER_ID else None

    def set_moderation(self, post_id: int, status: str, reasons: list[str],
                       detected_species: str | None) -> None:
        values = {":s": status, ":r": json.dumps(reasons)}
        expr = "SET #st = :s, reasons = :r"
        if detected_species is not None:
            expr += ", detected_species = :d"
            values[":d"] = detected_species
        else:
            expr += " REMOVE detected_species"
        self.table.update_item(
            Key={"id": post_id}, UpdateExpression=expr, ExpressionAttributeNames={"#st": "status"},
            ExpressionAttributeValues=values, ConditionExpression="attribute_exists(id)")

    def set_status_if(self, post_id: int, expected: str, new: str,
                      human: bool = False) -> bool:
        from botocore.exceptions import ClientError

        expr, values = "SET #st = :new", {":new": new, ":exp": expected}
        if human:
            expr += ", human_reviewed = :h"
            values[":h"] = True
        try:
            self.table.update_item(
                Key={"id": post_id}, UpdateExpression=expr,
                ConditionExpression="#st = :exp", ExpressionAttributeNames={"#st": "status"},
                ExpressionAttributeValues=values)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                return False
            raise

    def list_posts(self, status: str, species: str | None = None, q: str | None = None,
                   limit: int = 100) -> list[dict]:
        out: list[dict] = []
        kwargs = {"IndexName": GSI, "ScanIndexForward": False,
                  "KeyConditionExpression": "#st = :s",
                  "ExpressionAttributeNames": {"#st": "status"},
                  "ExpressionAttributeValues": {":s": status}}
        while len(out) < limit:
            page = self.table.query(**kwargs)
            for item in page["Items"]:
                row = self._row(item)
                if species and (row["detected_species"] or row["declared_species"]) != species:
                    continue
                if q and q.lower() not in (row["title"] + " " + row["description"]).lower():
                    continue
                out.append(row)
                if len(out) == limit:
                    break
            if "LastEvaluatedKey" not in page:
                break
            kwargs["ExclusiveStartKey"] = page["LastEvaluatedKey"]
        return out

    @staticmethod
    def _row(item: dict) -> dict:
        d = dict(item)
        d["id"] = int(d["id"])
        d["reasons"] = json.loads(d["reasons"])
        d.setdefault("detected_species", None)
        d["human_reviewed"] = bool(d.get("human_reviewed", False))
        return d
