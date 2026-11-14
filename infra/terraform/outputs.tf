output "bucket" { value = aws_s3_bucket.media.id }
output "table" { value = aws_dynamodb_table.posts.name }
output "ecr_repository" { value = aws_ecr_repository.worker.repository_url }
output "function" { value = aws_lambda_function.moderate.function_name }
