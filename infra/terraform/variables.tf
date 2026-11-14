variable "project" {
  type    = string
  default = "pawmatch-guard"
}

variable "region" {
  type    = string
  default = "ap-northeast-1"
}

variable "image_tag" {
  description = "Tag of the worker image in ECR. Build for the architecture below."
  type        = string
  default     = "latest"
}

variable "cpu_architecture" {
  description = "X86_64 or ARM64. Must match the architecture the image was built for."
  type        = string
  default     = "X86_64"
}

variable "worker_cpu" {
  type    = number
  default = 2048
}

variable "worker_memory" {
  type    = number
  default = 4096
}

variable "batch_max_vcpus" {
  description = "Upper bound on the re-evaluation job fleet (cost guard)."
  type        = number
  default     = 4
}

variable "upload_ia_days" {
  description = "Move raw uploads to Infrequent Access after this many days."
  type        = number
  default     = 30
}
