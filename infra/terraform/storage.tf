# --- S3: uploads/ (raw media), public/ (blurred output), models/ (weights) -------------
resource "aws_s3_bucket" "media" {
  bucket_prefix = "${var.project}-"
}

resource "aws_s3_bucket_public_access_block" "media" {
  bucket                  = aws_s3_bucket.media.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "media" {
  bucket = aws_s3_bucket.media.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "AES256" }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "media" {
  bucket = aws_s3_bucket.media.id

  rule {
    id     = "uploads-to-infrequent-access"
    status = "Enabled"
    filter { prefix = "uploads/" }
    transition {
      days          = var.upload_ia_days
      storage_class = "STANDARD_IA"
    }
  }

  rule {
    id     = "abort-incomplete-multipart"
    status = "Enabled"
    filter {}
    abort_incomplete_multipart_upload { days_after_initiation = 7 }
  }
}

# --- DynamoDB: posts (same schema as src/storage/dynamo.py:create_table) ---------------
resource "aws_dynamodb_table" "posts" {
  name         = "${var.project}-posts"
  billing_mode = "PROVISIONED" # on-demand tables get NO free read/write capacity
  hash_key     = "id"

  # Always-free tier: 25 provisioned WCU + 25 RCU per account and region, shared by all
  # tables. 5+5 for the table and 5+5 for the index uses 10 of each.
  read_capacity  = 5
  write_capacity = 5

  attribute {
    name = "id"
    type = "N"
  }
  attribute {
    name = "status"
    type = "S"
  }

  global_secondary_index {
    name            = "status-id-index"
    hash_key        = "status"
    range_key       = "id"
    projection_type = "ALL"
    read_capacity   = 5
    write_capacity  = 5
  }
}

# --- ECR: Lambda image. Free storage is 500 MB for the first 12 months only; a PyTorch image
# is larger than that, so expect about $0.10 per GB-month. Keep only one image to limit it.
resource "aws_ecr_repository" "worker" {
  name                 = "${var.project}-worker"
  image_tag_mutability = "MUTABLE"
  force_delete         = true
  image_scanning_configuration { scan_on_push = true }
}

resource "aws_ecr_lifecycle_policy" "worker" {
  repository = aws_ecr_repository.worker.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep only the latest image"
      selection    = { tagStatus = "any", countType = "imageCountMoreThan", countNumber = 1 }
      action       = { type = "expire" }
    }]
  })
}
