# ────────────────────────────────────────────────────────────────────────────────
# terraform/variables.tf
# Input variable definitions for the EKS cluster module
# ────────────────────────────────────────────────────────────────────────────────

variable "aws_region" {
  description = "AWS region to deploy the EKS cluster and supporting resources."
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Short identifier used to prefix all resource names (no spaces, lowercase)."
  type        = string
  default     = "sql-llmops"
}

variable "environment" {
  description = "Deployment environment tag (e.g. dev / staging / prod)."
  type        = string
  default     = "prod"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "environment must be one of: dev, staging, prod."
  }
}

# ── VPC / Networking ──────────────────────────────────────────────────────────

variable "vpc_cidr" {
  description = "CIDR block for the new VPC."
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "List of Availability Zones to span subnets across (min 2 for HA)."
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b", "us-east-1c"]
}

variable "private_subnet_cidrs" {
  description = "CIDR blocks for private subnets (one per AZ) — EKS node groups run here."
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for public subnets (one per AZ) — NAT Gateways and ALBs."
  type        = list(string)
  default     = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
}

# ── EKS Cluster ───────────────────────────────────────────────────────────────

variable "cluster_version" {
  description = "Kubernetes version for the EKS control plane."
  type        = string
  default     = "1.30"
}

variable "cluster_endpoint_public_access" {
  description = "Whether the EKS API server endpoint is publicly accessible."
  type        = bool
  default     = true
}

variable "cluster_endpoint_public_access_cidrs" {
  description = "List of CIDRs that can reach the public API endpoint. Restrict to your office / CI IPs."
  type        = list(string)
  default     = ["0.0.0.0/0"]   # ⚠ Tighten this for production!
}

# ── Node Groups ───────────────────────────────────────────────────────────────

variable "cpu_node_instance_types" {
  description = "EC2 instance types for the general-purpose CPU node group."
  type        = list(string)
  default     = ["m5.xlarge"]
}

variable "cpu_node_desired" {
  description = "Desired number of nodes in the CPU node group."
  type        = number
  default     = 2
}

variable "cpu_node_min" {
  description = "Minimum number of nodes in the CPU node group."
  type        = number
  default     = 1
}

variable "cpu_node_max" {
  description = "Maximum number of nodes in the CPU node group."
  type        = number
  default     = 5
}

variable "gpu_node_instance_types" {
  description = "EC2 instance types for the GPU node group (for model inference)."
  type        = list(string)
  default     = ["g4dn.xlarge"]   # 1× NVIDIA T4 GPU, 4 vCPU, 16 GiB RAM
}

variable "gpu_node_desired" {
  description = "Desired number of nodes in the GPU node group."
  type        = number
  default     = 1
}

variable "gpu_node_min" {
  description = "Minimum number of nodes in the GPU node group (0 = scale to zero)."
  type        = number
  default     = 0
}

variable "gpu_node_max" {
  description = "Maximum number of nodes in the GPU node group."
  type        = number
  default     = 3
}

# ── Tagging ───────────────────────────────────────────────────────────────────

variable "tags" {
  description = "Additional tags to apply to all resources."
  type        = map(string)
  default = {
    Project   = "sql-llmops-pipeline"
    ManagedBy = "terraform"
  }
}
