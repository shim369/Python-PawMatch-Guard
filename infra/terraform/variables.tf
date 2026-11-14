variable "project" {
  type    = string
  default = "pawmatch-guard"
}

variable "region" {
  type    = string
  default = "ap-northeast-1"
}

variable "image_tag" {
  description = "Tag of the Lambda container image in ECR."
  type        = string
  default     = "latest"
}

variable "lambda_architecture" {
  description = "arm64 (Graviton) or x86_64; must match how the image was built."
  type        = string
  default     = "arm64"
}

variable "lambda_memory_mb" {
  description = "More memory = more CPU but more GB-seconds per post (free tier: 400,000 GB-s/month)."
  type        = number
  default     = 3072
}

variable "alert_email" {
  description = "Receives a warning when monthly cost approaches the budget."
  type        = string
}

variable "monthly_budget_usd" {
  type    = number
  default = 1
}

variable "upload_ia_days" {
  description = "Move raw uploads to Infrequent Access after this many days."
  type        = number
  default     = 30
}
