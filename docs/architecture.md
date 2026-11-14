# Architecture / 構成

```mermaid
flowchart LR
  U[User] -->|upload| APP[FastAPI app]
  APP -->|row: pending| DDB[(DynamoDB<br/>provisioned 5/5)]
  APP -->|uploads/ID.ext| S3[(S3)]
  S3 -->|ObjectCreated| L[Lambda container<br/>classifier + detector + rules]
  L -->|weights, once per cold start| S3
  L -->|public/ID.jpg blurred| S3
  L -->|status, reasons| DDB
  R[Reviewer] -->|review queue| APP
  APP -->|human decision| DDB
  OP[Operator] -.->|aws lambda invoke<br/>action=reevaluate| L
  L -.->|chains itself, 20 posts per call| L
```

- **Only published posts are shown.** `review` posts wait for a person; `rejected` and `error`
  are hidden. 公開可のものだけが表示され、要確認は人の判断待ち。
- **Re-evaluation only tightens.** After a model update, published → review/rejected is
  allowed; nothing is ever loosened, and posts a person decided are never touched.
  再評価は厳しくする方向のみ。人が判断した投稿には触れない。
- **No always-on compute.** Nothing runs between uploads (no ECS service, no Batch fleet).
  アップロードがないとき、動いているものはありません。
- **Local equivalents** (`docker compose --profile aws-local`): S3 → SeaweedFS, DynamoDB →
  DynamoDB Local, Lambda → the same handler code. MinIO's official images can no longer be
  pulled, hence SeaweedFS.

The FastAPI app itself is *not* deployed by `infra/terraform`; run it anywhere with
`PAWMATCH_STORAGE=aws PAWMATCH_MODERATION=external` and the table/bucket names from the
Terraform outputs. アプリ本体は Terraform の対象外です。
