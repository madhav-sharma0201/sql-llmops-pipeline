# ────────────────────────────────────────────────────────────────────────────────
# terraform/outputs.tf
# Exported values for use by downstream scripts, CI pipelines, and kubectl
# ────────────────────────────────────────────────────────────────────────────────

# ── VPC ───────────────────────────────────────────────────────────────────────
output "vpc_id" {
  description = "ID of the created VPC."
  value       = aws_vpc.main.id
}

output "vpc_cidr" {
  description = "CIDR block of the VPC."
  value       = aws_vpc.main.cidr_block
}

output "public_subnet_ids" {
  description = "IDs of the public subnets (one per AZ)."
  value       = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  description = "IDs of the private subnets where EKS nodes run."
  value       = aws_subnet.private[*].id
}

output "nat_gateway_public_ip" {
  description = "Public IP of the NAT Gateway (add to firewall allowlists)."
  value       = aws_eip.nat.public_ip
}

# ── EKS Cluster ───────────────────────────────────────────────────────────────
output "cluster_name" {
  description = "Name of the EKS cluster."
  value       = aws_eks_cluster.main.name
}

output "cluster_arn" {
  description = "ARN of the EKS cluster."
  value       = aws_eks_cluster.main.arn
}

output "cluster_endpoint" {
  description = "Kubernetes API server endpoint URL."
  value       = aws_eks_cluster.main.endpoint
}

output "cluster_version" {
  description = "Kubernetes version of the EKS control plane."
  value       = aws_eks_cluster.main.version
}

output "cluster_certificate_authority_data" {
  description = "Base64-encoded certificate authority data for kubectl."
  value       = aws_eks_cluster.main.certificate_authority[0].data
  sensitive   = true
}

output "cluster_security_group_id" {
  description = "ID of the cluster security group created by EKS."
  value       = aws_eks_cluster.main.vpc_config[0].cluster_security_group_id
}

# ── Node Groups ───────────────────────────────────────────────────────────────
output "cpu_node_group_arn" {
  description = "ARN of the CPU managed node group."
  value       = aws_eks_node_group.cpu.arn
}

output "gpu_node_group_arn" {
  description = "ARN of the GPU managed node group."
  value       = aws_eks_node_group.gpu.arn
}

# ── IAM ───────────────────────────────────────────────────────────────────────
output "cluster_iam_role_arn" {
  description = "ARN of the IAM role used by the EKS control plane."
  value       = aws_iam_role.eks_cluster.arn
}

output "node_group_iam_role_arn" {
  description = "ARN of the IAM role used by EKS node groups."
  value       = aws_iam_role.node_group.arn
}

# ── OIDC ──────────────────────────────────────────────────────────────────────
output "oidc_provider_arn" {
  description = "ARN of the IAM OIDC provider (required for IRSA)."
  value       = aws_iam_openid_connect_provider.eks.arn
}

output "oidc_provider_url" {
  description = "URL of the IAM OIDC provider."
  value       = aws_iam_openid_connect_provider.eks.url
}

# ── KMS ───────────────────────────────────────────────────────────────────────
output "eks_kms_key_arn" {
  description = "ARN of the KMS key used for EKS Secrets encryption."
  value       = aws_kms_key.eks.arn
}

# ── Handy kubeconfig command ──────────────────────────────────────────────────
output "kubeconfig_command" {
  description = "Run this command to update your local kubeconfig."
  value       = "aws eks update-kubeconfig --region ${var.aws_region} --name ${aws_eks_cluster.main.name}"
}
