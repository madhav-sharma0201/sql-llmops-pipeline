# ────────────────────────────────────────────────────────────────────────────────
# terraform/main.tf
# AWS EKS Cluster with VPC, Subnets, NAT Gateway, IAM Roles, and Node Groups
#
# Resources created
# -----------------
#   VPC + Internet Gateway
#   Public & Private Subnets (multi-AZ)
#   NAT Gateway (single, for cost; add per-AZ for HA)
#   Route Tables + Associations
#   IAM Roles: EKS Cluster, Node Group, IRSA (OIDC)
#   EKS Control Plane
#   EKS Managed Node Groups: CPU (general) + GPU (inference)
#   aws-auth ConfigMap patch (maps IAM → Kubernetes RBAC)
#
# Usage
# -----
#   cd 3-infrastructure/terraform
#   terraform init
#   terraform plan -var-file=environments/prod.tfvars
#   terraform apply -var-file=environments/prod.tfvars
# ────────────────────────────────────────────────────────────────────────────────

terraform {
  required_version = ">= 1.6"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.50"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
  }

  # ── Remote state (recommended for teams) ────────────────────────────────────
  # backend "s3" {
  #   bucket         = "your-terraform-state-bucket"
  #   key            = "sql-llmops/terraform.tfstate"
  #   region         = "us-east-1"
  #   encrypt        = true
  #   dynamodb_table = "terraform-state-lock"
  # }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = merge(var.tags, {
      Environment = var.environment
      Project     = var.project_name
    })
  }
}

# ──────────────────────────────────────────────────────────────────────────────
# 1. DATA SOURCES
# ──────────────────────────────────────────────────────────────────────────────

# Fetch the current caller identity (for ARN composition)
data "aws_caller_identity" "current" {}

# Fetch available AZs in the target region
data "aws_availability_zones" "available" {
  state = "available"
}

# Latest EKS-optimised Amazon Linux 2 AMI (for node groups)
data "aws_ssm_parameter" "eks_ami" {
  name = "/aws/service/eks/optimized-ami/${var.cluster_version}/amazon-linux-2/recommended/image_id"
}

# Latest GPU-enabled AMI (Amazon Linux 2 with NVIDIA drivers)
data "aws_ssm_parameter" "eks_gpu_ami" {
  name = "/aws/service/eks/optimized-ami/${var.cluster_version}/amazon-linux-2-gpu/recommended/image_id"
}


# ──────────────────────────────────────────────────────────────────────────────
# 2. NETWORKING
# ──────────────────────────────────────────────────────────────────────────────

# ── VPC ───────────────────────────────────────────────────────────────────────
resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = { Name = "${var.project_name}-vpc" }
}

# ── Internet Gateway ──────────────────────────────────────────────────────────
resource "aws_internet_gateway" "igw" {
  vpc_id = aws_vpc.main.id
  tags   = { Name = "${var.project_name}-igw" }
}

# ── Public Subnets ────────────────────────────────────────────────────────────
resource "aws_subnet" "public" {
  count = length(var.public_subnet_cidrs)

  vpc_id                  = aws_vpc.main.id
  cidr_block              = var.public_subnet_cidrs[count.index]
  availability_zone       = var.availability_zones[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name                                              = "${var.project_name}-public-${count.index + 1}"
    "kubernetes.io/role/elb"                          = "1"   # Required by AWS LBC
    "kubernetes.io/cluster/${var.project_name}-eks"   = "shared"
  }
}

# ── Private Subnets ───────────────────────────────────────────────────────────
resource "aws_subnet" "private" {
  count = length(var.private_subnet_cidrs)

  vpc_id            = aws_vpc.main.id
  cidr_block        = var.private_subnet_cidrs[count.index]
  availability_zone = var.availability_zones[count.index]

  tags = {
    Name                                              = "${var.project_name}-private-${count.index + 1}"
    "kubernetes.io/role/internal-elb"                 = "1"   # Required by AWS LBC
    "kubernetes.io/cluster/${var.project_name}-eks"   = "shared"
  }
}

# ── NAT Gateway (single for dev/staging; add per-AZ for prod HA) ──────────────
resource "aws_eip" "nat" {
  domain     = "vpc"
  depends_on = [aws_internet_gateway.igw]
  tags       = { Name = "${var.project_name}-nat-eip" }
}

resource "aws_nat_gateway" "main" {
  allocation_id = aws_eip.nat.id
  subnet_id     = aws_subnet.public[0].id
  depends_on    = [aws_internet_gateway.igw]
  tags          = { Name = "${var.project_name}-nat-gw" }
}

# ── Route Tables ──────────────────────────────────────────────────────────────
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.igw.id
  }
  tags = { Name = "${var.project_name}-public-rt" }
}

resource "aws_route_table_association" "public" {
  count          = length(aws_subnet.public)
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table" "private" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.main.id
  }
  tags = { Name = "${var.project_name}-private-rt" }
}

resource "aws_route_table_association" "private" {
  count          = length(aws_subnet.private)
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private.id
}


# ──────────────────────────────────────────────────────────────────────────────
# 3. IAM — EKS CLUSTER ROLE
# ──────────────────────────────────────────────────────────────────────────────

resource "aws_iam_role" "eks_cluster" {
  name = "${var.project_name}-eks-cluster-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "eks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "eks_cluster_policy" {
  role       = aws_iam_role.eks_cluster.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
}

resource "aws_iam_role_policy_attachment" "eks_vpc_resource_controller" {
  role       = aws_iam_role.eks_cluster.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSVPCResourceController"
}


# ──────────────────────────────────────────────────────────────────────────────
# 4. EKS CONTROL PLANE
# ──────────────────────────────────────────────────────────────────────────────

resource "aws_eks_cluster" "main" {
  name     = "${var.project_name}-eks"
  version  = var.cluster_version
  role_arn = aws_iam_role.eks_cluster.arn

  vpc_config {
    subnet_ids              = concat(aws_subnet.private[*].id, aws_subnet.public[*].id)
    endpoint_public_access  = var.cluster_endpoint_public_access
    public_access_cidrs     = var.cluster_endpoint_public_access_cidrs
    endpoint_private_access = true
  }

  # Enable useful control-plane logging
  enabled_cluster_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]

  # Encryption at rest for Kubernetes Secrets
  encryption_config {
    provider {
      key_arn = aws_kms_key.eks.arn
    }
    resources = ["secrets"]
  }

  depends_on = [
    aws_iam_role_policy_attachment.eks_cluster_policy,
    aws_iam_role_policy_attachment.eks_vpc_resource_controller,
  ]

  tags = { Name = "${var.project_name}-eks" }
}

# KMS key for envelope encryption of Kubernetes Secrets
resource "aws_kms_key" "eks" {
  description             = "EKS Secrets encryption key for ${var.project_name}"
  deletion_window_in_days = 10
  enable_key_rotation     = true
  tags                    = { Name = "${var.project_name}-eks-kms" }
}


# ──────────────────────────────────────────────────────────────────────────────
# 5. IAM — NODE GROUP ROLE
# ──────────────────────────────────────────────────────────────────────────────

resource "aws_iam_role" "node_group" {
  name = "${var.project_name}-eks-node-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

locals {
  node_policies = [
    "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy",
    "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy",
    "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly",
    "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore",   # for SSM Session Manager access
  ]
}

resource "aws_iam_role_policy_attachment" "node_policies" {
  count      = length(local.node_policies)
  role       = aws_iam_role.node_group.name
  policy_arn = local.node_policies[count.index]
}


# ──────────────────────────────────────────────────────────────────────────────
# 6. EKS MANAGED NODE GROUPS
# ──────────────────────────────────────────────────────────────────────────────

# ── CPU Node Group (general workloads: ArgoCD, API pods on CPU) ───────────────
resource "aws_eks_node_group" "cpu" {
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "${var.project_name}-cpu-ng"
  node_role_arn   = aws_iam_role.node_group.arn
  subnet_ids      = aws_subnet.private[*].id
  instance_types  = var.cpu_node_instance_types
  ami_type        = "AL2_x86_64"
  disk_size       = 50   # GiB

  scaling_config {
    desired_size = var.cpu_node_desired
    min_size     = var.cpu_node_min
    max_size     = var.cpu_node_max
  }

  update_config {
    max_unavailable = 1
  }

  # Use spot instances to reduce cost (falls back to on-demand automatically)
  capacity_type = "SPOT"

  labels = {
    role        = "general"
    node-type   = "cpu"
  }

  depends_on = [aws_iam_role_policy_attachment.node_policies]
  tags       = { Name = "${var.project_name}-cpu-node-group" }
}

# ── GPU Node Group (inference workloads: Llama model serving) ─────────────────
resource "aws_eks_node_group" "gpu" {
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "${var.project_name}-gpu-ng"
  node_role_arn   = aws_iam_role.node_group.arn
  subnet_ids      = aws_subnet.private[*].id
  instance_types  = var.gpu_node_instance_types
  ami_type        = "AL2_x86_64_GPU"   # Amazon Linux 2 with NVIDIA drivers pre-installed
  disk_size       = 100  # GiB (model weights + swap)

  scaling_config {
    desired_size = var.gpu_node_desired
    min_size     = var.gpu_node_min
    max_size     = var.gpu_node_max
  }

  update_config {
    max_unavailable = 1
  }

  capacity_type = "ON_DEMAND"   # GPU spot is less available — use on-demand for reliability

  labels = {
    role        = "inference"
    node-type   = "gpu"
    accelerator = "nvidia"
  }

  # Taint GPU nodes so only GPU-requesting pods are scheduled here
  taint {
    key    = "nvidia.com/gpu"
    value  = "true"
    effect = "NO_SCHEDULE"
  }

  depends_on = [aws_iam_role_policy_attachment.node_policies]
  tags       = { Name = "${var.project_name}-gpu-node-group" }
}


# ──────────────────────────────────────────────────────────────────────────────
# 7. OIDC PROVIDER (required for IRSA — IAM Roles for Service Accounts)
# ──────────────────────────────────────────────────────────────────────────────

data "tls_certificate" "eks_oidc" {
  url = aws_eks_cluster.main.identity[0].oidc[0].issuer
}

resource "aws_iam_openid_connect_provider" "eks" {
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = [data.tls_certificate.eks_oidc.certificates[0].sha1_fingerprint]
  url             = aws_eks_cluster.main.identity[0].oidc[0].issuer
}


# ──────────────────────────────────────────────────────────────────────────────
# 8. EKS ADD-ONS
# ──────────────────────────────────────────────────────────────────────────────

resource "aws_eks_addon" "vpc_cni" {
  cluster_name = aws_eks_cluster.main.name
  addon_name   = "vpc-cni"
}

resource "aws_eks_addon" "coredns" {
  cluster_name = aws_eks_cluster.main.name
  addon_name   = "coredns"
  depends_on   = [aws_eks_node_group.cpu]   # CoreDNS needs at least one node
}

resource "aws_eks_addon" "kube_proxy" {
  cluster_name = aws_eks_cluster.main.name
  addon_name   = "kube-proxy"
}

resource "aws_eks_addon" "ebs_csi_driver" {
  cluster_name = aws_eks_cluster.main.name
  addon_name   = "aws-ebs-csi-driver"
  # Service account annotation for IRSA would go in a separate IAM role resource
}
