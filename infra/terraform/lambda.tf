data "archive_file" "trigger" {
  type        = "zip"
  source_file = "${path.module}/../lambda/handler.py"
  output_path = "${path.module}/.build/trigger.zip"
}

data "aws_iam_policy_document" "lambda_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "trigger" {
  name_prefix        = "${var.project}-trigger-"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

resource "aws_iam_role_policy_attachment" "trigger_logs" {
  role       = aws_iam_role.trigger.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

data "aws_iam_policy_document" "trigger" {
  statement {
    actions   = ["ecs:RunTask"]
    resources = [aws_ecs_task_definition.worker.arn]
    condition {
      test     = "ArnEquals"
      variable = "ecs:cluster"
      values   = [aws_ecs_cluster.main.arn]
    }
  }
  statement {
    actions   = ["iam:PassRole"]
    resources = [aws_iam_role.task.arn, aws_iam_role.task_execution.arn]
  }
}

resource "aws_iam_role_policy" "trigger" {
  role   = aws_iam_role.trigger.id
  policy = data.aws_iam_policy_document.trigger.json
}

resource "aws_lambda_function" "trigger" {
  function_name    = "${var.project}-trigger"
  role             = aws_iam_role.trigger.arn
  runtime          = "python3.12"
  handler          = "handler.handler"
  filename         = data.archive_file.trigger.output_path
  source_code_hash = data.archive_file.trigger.output_base64sha256
  timeout          = 30

  environment {
    variables = {
      CLUSTER_ARN         = aws_ecs_cluster.main.arn
      TASK_DEFINITION_ARN = aws_ecs_task_definition.worker.arn
      SUBNET_IDS          = join(",", data.aws_subnets.default.ids)
      SECURITY_GROUP_ID   = aws_security_group.worker.id
    }
  }
}

resource "aws_lambda_permission" "from_s3" {
  statement_id  = "AllowS3Invoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.trigger.function_name
  principal     = "s3.amazonaws.com"
  source_arn    = aws_s3_bucket.media.arn
}

resource "aws_s3_bucket_notification" "uploads" {
  bucket = aws_s3_bucket.media.id
  lambda_function {
    lambda_function_arn = aws_lambda_function.trigger.arn
    events              = ["s3:ObjectCreated:*"]
    filter_prefix       = "uploads/"
  }
  depends_on = [aws_lambda_permission.from_s3]
}
