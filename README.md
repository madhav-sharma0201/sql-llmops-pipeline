# Automated LLMOps Pipeline — Specialized Text-to-SQL Generator

> **Portfolio Project** · MLOps · Fine-Tuning · FastAPI · Kubernetes · GitOps

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)](https://fastapi.tiangolo.com)
[![Terraform](https://img.shields.io/badge/Terraform-1.6%2B-purple)](https://terraform.io)
[![ArgoCD](https://img.shields.io/badge/ArgoCD-GitOps-orange)](https://argo-cd.readthedocs.io)

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  TRAINING ENVIRONMENT (Google Colab / Local GPU)                │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │ sql-create- │  │ dataset_     │  │ train_qlora.py        │  │
│  │ context     │→ │ formatter.py │→ │ Unsloth + QLoRA + TRL │  │
│  │ (78k rows)  │  │              │  │ Llama-3.2-3B-Instruct │  │
│  └─────────────┘  └──────────────┘  └──────────┬────────────┘  │
│                                                  │ adapter_weights/
│                                    ┌─────────────▼────────────┐  │
│                                    │ MLflow Experiment Tracker │  │
│                                    └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                            │ push weights to HF Hub / S3
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  INFERENCE LAYER (Docker → EKS)                                 │
│  ┌──────────────────────────────┐                               │
│  │  FastAPI  main.py            │                               │
│  │  POST /v1/generate-sql       │ ← REST clients                │
│  │  GET  /health                │                               │
│  └──────────────────────────────┘                               │
│  Dockerised → pushed to ECR → deployed via ArgoCD GitOps        │
└─────────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────────┐
│  INFRASTRUCTURE (Terraform + Kubernetes + ArgoCD)               │
│  AWS EKS (VPC, Private Subnets, CPU+GPU Node Groups)            │
│  ArgoCD watches Git → syncs K8s manifests automatically         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
sql-llmops-pipeline/
│
├── 📁 1-model-training/          # QLoRA Fine-Tuning
│   ├── train_qlora.py            # Main training script (Unsloth + TRL)
│   ├── evaluate_metrics.py       # Base vs fine-tuned evaluation
│   ├── mlflow_tracker.py         # MLflow experiment tracking
│   └── dataset_formatter.py      # sql-create-context → instruction format
│
├── 📁 2-inference-api/           # Serving Layer
│   ├── main.py                   # FastAPI server
│   ├── Dockerfile                # Multi-stage production container
│   ├── requirements.txt          # Pinned Python dependencies
│   └── .env.example              # Environment variable template
│
├── 📁 3-infrastructure/          # IaC + GitOps
│   ├── terraform/
│   │   ├── main.tf               # EKS cluster, VPC, node groups
│   │   ├── variables.tf          # Input variables
│   │   └── outputs.tf            # Exported resource identifiers
│   └── k8s-manifests/
│       ├── deployment.yaml       # Pod spec, probes, resource limits
│       ├── service.yaml          # ClusterIP + ALB Ingress + HPA
│       └── argocd-app.yaml       # GitOps sync controller
│
├── 📁 4-frontend/                # Web Workbench & MLOps UI
│   ├── index.html                # Interactive Glassmorphism UI
│   ├── styles.css                # Dark mode HSL Design System
│   └── app.js                    # Preset schemas, API fetch & Mock fallback
│
├── docker-compose.yml            # Local dev stack
├── .gitignore
└── README.md
```

---

## ⚡ Quick Start

### Step 1 — Fine-tune the Model (Google Colab / Local GPU)

```bash
# Install dependencies
pip install unsloth trl peft bitsandbytes transformers accelerate \
            mlflow sacrebleu datasets evaluate

# Run training
cd 1-model-training
python train_qlora.py
# Adapter weights saved to: ./adapter_weights/

# Evaluate: base model vs fine-tuned
python evaluate_metrics.py \
  --adapter_path ./adapter_weights \
  --n_samples 100 \
  --output_json results/eval_report.json
```

### Step 2 — Run the Inference API Locally

```bash
# Using Docker Compose (recommended)
docker compose up --build api

# Or run directly
cd 2-inference-api
pip install -r requirements.txt
MODEL_PATH=../1-model-training/adapter_weights \
  uvicorn main:app --host 0.0.0.0 --port 8000
```

Test the API:

```bash
curl -X POST http://localhost:8000/v1/generate-sql \
  -H "Content-Type: application/json" \
  -d '{
    "schema": "CREATE TABLE employees (id INT, name TEXT, department TEXT, salary DECIMAL);",
    "question": "Find all employees in Engineering with salary over 80000"
  }'
```

Expected response:

```json
{
  "sql": "SELECT * FROM employees WHERE department = 'Engineering' AND salary > 80000;",
  "execution_time_ms": 342.5,
  "model_version": "v1.0.0",
  "tokens_generated": 18
}
```

### Step 3 — Deploy to AWS EKS

#### 3a. Provision Infrastructure

```bash
cd 3-infrastructure/terraform
terraform init
terraform plan -out=tfplan
terraform apply tfplan

# Configure kubectl
$(terraform output -raw kubeconfig_command)
```

#### 3b. Install ArgoCD

```bash
kubectl create namespace argocd
kubectl apply -n argocd \
  -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# Wait for ArgoCD to be ready
kubectl rollout status deployment/argocd-server -n argocd
```

#### 3c. Register the GitOps Application

```bash
# Update argocd-app.yaml with your actual Git repo URL, then:
kubectl apply -f 3-infrastructure/k8s-manifests/argocd-app.yaml -n argocd

# Watch sync status
argocd app get sql-llmops-pipeline
```

From this point, every `git push` to the `main` branch will automatically deploy the updated manifests to your EKS cluster.

---

## 🧪 Evaluation Benchmarks

| Metric                | Base Model | Fine-tuned | Δ Improvement |
|-----------------------|-----------|------------|---------------|
| Exact Match Accuracy  | ~12 %     | ~68 %      | **+56 pp**    |
| Execution Accuracy    | ~18 %     | ~74 %      | **+56 pp**    |
| BLEU-4                | 8.2       | 52.1       | **+43.9**     |
| Avg. Latency (ms)     | —         | ~340 ms    | —             |

> *Results on 100-sample held-out test split. Your numbers may vary based on training time and GPU.*

---

## 🔑 API Reference

### `GET /health`

Returns service status and model metadata.

```json
{
  "status": "healthy",
  "model_version": "v1.0.0",
  "model_path": "./adapter_weights",
  "uptime_seconds": 3621.4,
  "device": "cuda:0"
}
```

### `POST /v1/generate-sql`

**Request body:**

| Field              | Type   | Required | Description                              |
|--------------------|--------|----------|------------------------------------------|
| `schema`           | string | ✅       | SQL CREATE TABLE statement(s)            |
| `question`         | string | ✅       | Natural language question                |
| `max_new_tokens`   | int    | ❌       | Override token budget (1–512)            |
| `temperature`      | float  | ❌       | Sampling temperature (0.0 = greedy)      |

**Response:**

| Field                | Type   | Description                           |
|----------------------|--------|---------------------------------------|
| `sql`                | string | Generated SQL query                   |
| `execution_time_ms`  | float  | Inference latency in milliseconds     |
| `model_version`      | string | Deployed model version tag            |
| `tokens_generated`   | int    | Number of tokens produced             |

---

## 🛠️ Tech Stack

| Layer              | Technology                                |
|--------------------|-------------------------------------------|
| Fine-tuning        | Unsloth, PEFT, TRL, HuggingFace           |
| Base Model         | Llama-3.2-3B-Instruct (4-bit QLoRA)       |
| Dataset            | b-mc2/sql-create-context (78K pairs)      |
| Experiment Tracking| MLflow                                    |
| Inference API      | FastAPI + Uvicorn                         |
| Containerisation   | Docker (multi-stage)                      |
| Orchestration      | Kubernetes (AWS EKS)                      |
| GitOps             | ArgoCD                                    |
| Infrastructure     | Terraform (AWS VPC, EKS, IAM, KMS)        |
| Local Dev          | Docker Compose                            |

---

## 📜 License

MIT © 2024 — See [LICENSE](LICENSE) for details.
