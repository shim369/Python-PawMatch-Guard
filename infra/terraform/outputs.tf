output "bucket" { value = aws_s3_bucket.media.id }
output "table" { value = aws_dynamodb_table.posts.name }
output "ecr_repository" { value = aws_ecr_repository.worker.repository_url }
output "cluster" { value = aws_ecs_cluster.main.name }
output "reevaluate_queue" { value = aws_batch_job_queue.reevaluate.name }
output "reevaluate_job_definition" { value = aws_batch_job_definition.reevaluate.name }
