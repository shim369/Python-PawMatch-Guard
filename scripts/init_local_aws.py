"""Create the S3 bucket (SeaweedFS) and DynamoDB Local table of the `aws-local` profile.
Idempotent. Reads the same PAWMATCH_* variables as the app."""

import os

import boto3
from botocore.exceptions import ClientError

from src.storage.dynamo import create_table

bucket, table = os.environ["PAWMATCH_BUCKET"], os.environ["PAWMATCH_TABLE"]
s3 = boto3.client("s3", endpoint_url=os.environ["PAWMATCH_S3_ENDPOINT"])
try:
    s3.create_bucket(Bucket=bucket)
except ClientError as e:
    if e.response["Error"]["Code"] not in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
        raise
ddb = boto3.client("dynamodb", endpoint_url=os.environ["PAWMATCH_DYNAMODB_ENDPOINT"])
try:
    create_table(ddb, table)
except ClientError as e:
    if e.response["Error"]["Code"] != "ResourceInUseException":
        raise
print("ready:", bucket, table)
