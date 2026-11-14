# One container Lambda does the inference: S3 upload -> moderate that post; and a manual
# {"action":"reevaluate"} invocation re-checks live posts after a model update (it chains
# itself in chunks). No ECS/Fargate/Batch: those have no free tier.
locals {
  function_name = "${var.project}-moderate"
  function_arn  = "arn:aws:lambda:${var.region}:${data.aws_caller_identity.current.account_id}:function:${local.function_name}"
}

data "aws_caller_identity" "current" {}

data "aws_iam_policy_document" "lambda_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "moderate" {
  name_prefix        = "${var.project}-moderate-"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

resource "aws_iam_role_policy_attachment" "logs" {
  role       = aws_iam_role.moderate.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# Least privilege: this bucket, this table, and itself (for the re-evaluation chain).
data "aws_iam_policy_document" "moderate" {
  statement {
    actions   = ["s3:GetObject", "s3:PutObject"]
    resources = ["${aws_s3_bucket.media.arn}/*"]
  }
  statement {
    actions = ["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:UpdateItem", "dynamodb:Query"]
    resources = [
      aws_dynamodb_table.posts.arn,
      "${aws_dynamodb_table.posts.arn}/index/*",
    ]
  }
  statement {
    actions   = ["lambda:InvokeFunction"]
    resources = [local.function_arn]
  }
}

resource "aws_iam_role_policy" "moderate" {
  role   = aws_iam_role.moderate.id
  policy = data.aws_iam_policy_document.moderate.json
}

resource "aws_cloudwatch_log_group" "moderate" {
  name              = "/aws/lambda/${local.function_name}"
  retention_in_days = 7
}

resource "aws_lambda_function" "moderate" {
  function_name = local.function_name
  role          = aws_iam_role.moderate.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.worker.repository_url}:${var.image_tag}"
  architectures = [var.lambda_architecture]
  memory_size   = var.lambda_memory_mb
  timeout       = 600

  ephemeral_storage { size = 1024 }

  environment {
    variables = {
      PAWMATCH_STORAGE      = "aws"
      PAWMATCH_TABLE        = aws_dynamodb_table.posts.name
      PAWMATCH_BUCKET       = aws_s3_bucket.media.id
      PAWMATCH_MODEL_PREFIX = "models"
    }
  }

  depends_on = [aws_cloudwatch_log_group.moderate, aws_iam_role_policy_attachment.logs]
}

resource "aws_lambda_permission" "from_s3" {
  statement_id  = "AllowS3Invoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.moderate.function_name
  principal     = "s3.amazonaws.com"
  source_arn    = aws_s3_bucket.media.arn
}

resource "aws_s3_bucket_notification" "uploads" {
  bucket = aws_s3_bucket.media.id
  lambda_function {
    lambda_function_arn = aws_lambda_function.moderate.arn
    events              = ["s3:ObjectCreated:*"]
    filter_prefix       = "uploads/"
  }
  depends_on = [aws_lambda_permission.from_s3]
}
