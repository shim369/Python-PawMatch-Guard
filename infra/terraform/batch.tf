# Bulk re-evaluation after a model update: `pawmatch reevaluate` on Fargate Spot.
# The compute environment scales to zero when the queue is empty.
data "aws_iam_policy_document" "batch_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["batch.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "batch_service" {
  name_prefix        = "${var.project}-batch-"
  assume_role_policy = data.aws_iam_policy_document.batch_assume.json
}

resource "aws_iam_role_policy_attachment" "batch_service" {
  role       = aws_iam_role.batch_service.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSBatchServiceRole"
}

resource "aws_batch_compute_environment" "spot" {
  compute_environment_name_prefix = "${var.project}-"
  type                            = "MANAGED"
  service_role                    = aws_iam_role.batch_service.arn

  compute_resources {
    type               = "FARGATE_SPOT"
    max_vcpus          = var.batch_max_vcpus
    subnets            = data.aws_subnets.default.ids
    security_group_ids = [aws_security_group.worker.id]
  }

  depends_on = [aws_iam_role_policy_attachment.batch_service]
  lifecycle { create_before_destroy = true }
}

resource "aws_batch_job_queue" "reevaluate" {
  name     = "${var.project}-reevaluate"
  state    = "ENABLED"
  priority = 1
  compute_environment_order {
    order               = 1
    compute_environment = aws_batch_compute_environment.spot.arn
  }
}

resource "aws_batch_job_definition" "reevaluate" {
  name                  = "${var.project}-reevaluate"
  type                  = "container"
  platform_capabilities = ["FARGATE"]

  retry_strategy { attempts = 2 }
  timeout { attempt_duration_seconds = 7200 }

  container_properties = jsonencode({
    image                = local.worker_image
    command              = ["reevaluate"]
    environment          = local.worker_environment
    jobRoleArn           = aws_iam_role.task.arn
    executionRoleArn     = aws_iam_role.task_execution.arn
    runtimePlatform      = { operatingSystemFamily = "LINUX", cpuArchitecture = var.cpu_architecture }
    networkConfiguration = { assignPublicIp = "ENABLED" }
    resourceRequirements = [
      { type = "VCPU", value = tostring(var.worker_cpu / 1024) },
      { type = "MEMORY", value = tostring(var.worker_memory) },
    ]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.worker.name
        awslogs-region        = var.region
        awslogs-stream-prefix = "reevaluate"
      }
    }
  })
}
