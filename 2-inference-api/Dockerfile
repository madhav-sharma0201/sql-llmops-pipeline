# ────────────────────────────────────────────────────────────────────────────────
# Multi-stage Dockerfile for the Text-to-SQL FastAPI Inference Service
# ────────────────────────────────────────────────────────────────────────────────
#
# Stage 1  (builder)  — install Python dependencies into a virtual env
# Stage 2  (runtime)  — copy only the venv + app code for a slim final image
#
# Build:
#   docker build -t sql-inference-api:latest .
#
# Run (CPU):
#   docker run --rm -p 8000:8000 \
#     -v $(pwd)/adapter_weights:/app/adapter_weights \
#     --env-file .env \
#     sql-inference-api:latest
#
# Run (GPU — requires NVIDIA Container Toolkit):
#   docker run --rm --gpus all -p 8000:8000 \
#     -v $(pwd)/adapter_weights:/app/adapter_weights \
#     --env-file .env \
#     sql-inference-api:latest
# ────────────────────────────────────────────────────────────────────────────────

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Avoid interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# System build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# Create and activate a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip first to avoid resolver issues
RUN pip install --upgrade pip setuptools wheel

# Copy only the requirements file so Docker can cache this layer
COPY 2-inference-api/requirements.txt .

# Install Python dependencies (no-cache to keep layer size small)
RUN pip install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Labels (OCI Image Spec)
LABEL org.opencontainers.image.title="sql-inference-api"
LABEL org.opencontainers.image.description="LLMOps Text-to-SQL FastAPI inference service"
LABEL org.opencontainers.image.source="https://github.com/your-org/sql-llmops-pipeline"

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime system libraries (libgomp for PyTorch, curl for health-checks)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

# Copy application source and frontend static files
COPY 2-inference-api/main.py .
COPY 4-frontend /4-frontend


# Adapter weights loaded dynamically or via Fast Engine
# COPY 1-model-training/adapter_weights ./adapter_weights

# Set ownership
RUN chown -R appuser:appuser /app
USER appuser

# ── Environment defaults (override via --env-file or -e flags) ────────────────
ENV MODEL_PATH=/app/adapter_weights
ENV BASE_MODEL=unsloth/Llama-3.2-3B-Instruct
ENV MODEL_VERSION=v1.0.0
ENV TEMPERATURE=0.0
ENV MAX_NEW_TOKENS=256
ENV PORT=8000
ENV LOG_LEVEL=info

# Expose the API port
EXPOSE ${PORT}

# Health-check (matches the /health FastAPI endpoint)
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Launch the Uvicorn server
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT} --log-level ${LOG_LEVEL} --workers 1"]
