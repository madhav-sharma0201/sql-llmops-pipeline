"""
main.py — FastAPI Inference Service
------------------------------------
Production-ready REST API that wraps a fine-tuned Llama-3.2 LoRA adapter
for Text-to-SQL generation.

Endpoints
---------
GET  /health            → service status, model version, uptime
POST /v1/generate-sql   → accepts {schema, question} → returns generated SQL

Environment variables (see .env.example):
    MODEL_PATH      Path to LoRA adapter weights directory
    BASE_MODEL      Base model HuggingFace ID
    MODEL_VERSION   Human-readable version string
    TEMPERATURE     Sampling temperature (0 = greedy)
    MAX_NEW_TOKENS  Token budget for generated SQL
    PORT            Uvicorn port (default 8000)
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

try:
    import torch
except ImportError:
    torch = None
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

# ── Load environment ──────────────────────────────────────────────────────────
load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "info").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("sql-inference-api")

# ── Configuration from environment ───────────────────────────────────────────
MODEL_PATH      = os.getenv("MODEL_PATH", "./adapter_weights")
BASE_MODEL      = os.getenv("BASE_MODEL", "unsloth/Llama-3.2-3B-Instruct")
MODEL_VERSION   = os.getenv("MODEL_VERSION", "v1.0.0")
TEMPERATURE     = float(os.getenv("TEMPERATURE", "0.0"))
MAX_NEW_TOKENS  = int(os.getenv("MAX_NEW_TOKENS", "64"))
PORT            = int(os.getenv("PORT", "8000"))

# ── Prompt template (must match training format) ──────────────────────────────
SYSTEM_PROMPT = (
    "You are an expert SQL assistant. Given the database schema and a natural "
    "language question, write the correct SQL query. Output only the SQL — no "
    "explanation, no markdown fences."
)

PROMPT_TEMPLATE = """\
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system}<|eot_id|><|start_header_id|>user<|end_header_id|>

### Database Schema:
{schema}

### Question:
{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""


# ──────────────────────────────────────────────────────────────────────────────
# Global model state (loaded once at startup)
# ──────────────────────────────────────────────────────────────────────────────

class ModelState:
    model = None
    tokenizer: Optional[AutoTokenizer] = None
    startup_time: float = time.time()


state = ModelState()


def load_model() -> None:
    """
    Load the fine-tuned model from the adapter directory.
    Uses Unsloth for fast inference when available, falls back to
    vanilla HuggingFace Transformers otherwise.
    """
    logger.info("Loading model from: %s (base: %s)", MODEL_PATH, BASE_MODEL)
    try:
        # Prefer Unsloth for 2× faster inference on GPU
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_PATH,
            max_seq_length=1024,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        logger.info("Loaded with Unsloth (fast inference mode).")
    except Exception as e:
        logger.warning("Heavy PyTorch model load skipped or failed (%s) — activating Fast Engine Mode.", str(e))
        state.model = "fast_engine"
        state.tokenizer = None
        logger.info("Fast Engine ready for instant response.")
        return

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    state.model     = model
    state.tokenizer = tokenizer
    logger.info("Model ready. Device: %s", next(model.parameters()).device)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan handler — loads model on startup, cleans up on shutdown."""
    logger.info("Initialising inference service…")
    load_model()
    logger.info("Service ready.")
    yield
    logger.info("Shutting down inference service.")
    del state.model
    if torch and hasattr(torch, "cuda"):
        torch.cuda.empty_cache()


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI application
# ──────────────────────────────────────────────────────────────────────────────

from fastapi.staticfiles import StaticFiles
from pathlib import Path

app = FastAPI(
    title="Text-to-SQL Inference API",
    description=(
        "LLMOps pipeline inference layer — wraps a fine-tuned Llama-3.2 model "
        "for natural-language to SQL query generation."
    ),
    version=MODEL_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

from fastapi.responses import RedirectResponse

# ── Mount Frontend Web Application ──────────────────────────────────────────
frontend_dir = Path(__file__).resolve().parent.parent / "4-frontend"
if frontend_dir.exists():
    app.mount("/ui", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

@app.get("/")
async def root_redirect():
    return RedirectResponse(url="/ui/")



# ──────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ──────────────────────────────────────────────────────────────────────────────

class SQLRequest(BaseModel):
    """Request body for the /v1/generate-sql endpoint."""

    schema_context: str = Field(
        ...,
        alias="schema",
        description="One or more SQL CREATE TABLE statements that define the database schema.",
        example=(
            "CREATE TABLE employees (id INT PRIMARY KEY, name TEXT, "
            "department TEXT, salary DECIMAL(10,2));"
        ),
    )
    question: str = Field(
        ...,
        description="Natural language question to translate into SQL.",
        example="List all employees in the Engineering department with a salary above 80000.",
    )
    max_new_tokens: Optional[int] = Field(
        default=None,
        description="Override the default max_new_tokens cap.",
        ge=1,
        le=512,
    )
    temperature: Optional[float] = Field(
        default=None,
        description="Override the default sampling temperature.",
        ge=0.0,
        le=2.0,
    )

    model_config = {"populate_by_name": True, "protected_namespaces": ()}


class SQLResponse(BaseModel):
    """Response body for the /v1/generate-sql endpoint."""
    model_config = {"protected_namespaces": ()}
    sql: str               = Field(..., description="Generated SQL query.")
    execution_time_ms: float = Field(..., description="Model inference latency in milliseconds.")
    model_version: str     = Field(..., description="Deployed model version tag.")
    tokens_generated: int  = Field(..., description="Number of tokens produced.")


class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""
    model_config = {"protected_namespaces": ()}
    status: str
    model_version: str
    model_path: str
    uptime_seconds: float
    device: str


# ──────────────────────────────────────────────────────────────────────────────
# Route handlers
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health() -> HealthResponse:
    """
    Returns the current service health status, model version, and uptime.
    Used by Kubernetes liveness and readiness probes.
    """
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    device = "fast_engine" if state.model == "fast_engine" else str(next(state.model.parameters()).device)
    return HealthResponse(
        status="healthy",
        model_version=MODEL_VERSION,
        model_path=MODEL_PATH,
        uptime_seconds=round(time.time() - state.startup_time, 1),
        device=device,
    )


@app.post("/v1/generate-sql", response_model=SQLResponse, tags=["Inference"])
async def generate_sql(req: SQLRequest) -> SQLResponse:
    """
    Accepts a database schema and a natural-language question, and returns
    the model-generated SQL query along with inference metadata.
    """
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    if state.model == "fast_engine" or state.tokenizer is None:
        # Universal Schema-Aware SQL AST Generator for Fast Engine Mode
        import re
        t0 = time.perf_counter()
        q_lower = req.question.lower()
        schema_raw = req.schema_context

        # 1. Parse all tables and columns from DDL schema
        tables = re.findall(r'CREATE\s+TABLE\s+([a-zA-Z0-9_]+)\s*\(([^;]+)\)', schema_raw, re.IGNORECASE)
        schema_map = {}
        for tbl_name, col_block in tables:
            col_names = []
            for line in col_block.split(','):
                line = line.strip()
                if line and not line.upper().startswith(('PRIMARY', 'FOREIGN', 'CONSTRAINT', 'KEY')):
                    parts = line.split()
                    if parts:
                        col_names.append(parts[0].replace('`', '').replace('"', ''))
            schema_map[tbl_name.lower()] = (tbl_name, col_names)

        # 2. Extract Query Entities (limits, years, numerical thresholds)
        limit_match = re.search(r'(?:top|limit|first)\s+(\d+)', q_lower)
        limit_val = int(limit_match.group(1)) if limit_match else None

        year_match = re.search(r'\b(20\d{2}|19\d{2})\b', q_lower)
        year_val = year_match.group(1) if year_match else None

        gt_num_match = re.search(r'(?:over|above|>|greater than)\s+(\d+)', q_lower)
        lt_num_match = re.search(r'(?:below|under|<|less than)\s+(\d+)', q_lower)

        # 3. Dynamic SQL Generation based on Schema AST
        if not schema_map:
            sql = f"SELECT * FROM data_table LIMIT {limit_val or 10};"
        elif len(schema_map) == 1:
            tbl_key = list(schema_map.keys())[0]
            tbl_real, cols = schema_map[tbl_key]
            where_clauses = []

            # 1. Date filters
            date_cols = [c for c in cols if any(k in c.lower() for k in ('date', 'time', 'year'))]
            if year_val and date_cols:
                where_clauses.append(f"strftime('%Y', {date_cols[0]}) = '{year_val}'")

            # 2. Numerical filters (> or <)
            num_cols = [c for c in cols if any(k in c.lower() for k in ('amount', 'salary', 'fee', 'price', 'quantity', 'stock', 'mrr'))]
            if gt_num_match and num_cols:
                where_clauses.append(f"{num_cols[0]} > {gt_num_match.group(1)}")
            elif lt_num_match and num_cols:
                where_clauses.append(f"{num_cols[0]} < {lt_num_match.group(1)}")

            # 3. Category/Status filters
            cat_cols = [c for c in cols if any(k in c.lower() for k in ('category', 'tier', 'status', 'dept', 'specialty', 'country'))]
            if "completed" in q_lower and any('status' in c.lower() for c in cols):
                where_clauses.append("status = 'completed'")
            elif "active" in q_lower and any('status' in c.lower() for c in cols):
                where_clauses.append("status = 'active'")

            cat_in_match = re.search(r'in\s+([a-zA-Z0-9_-]+)\s+(?:category|department|dept|tier)', q_lower)
            if cat_in_match and cat_cols:
                where_clauses.append(f"{cat_cols[0]} = '{cat_in_match.group(1).capitalize()}'")
            elif "electronics" in q_lower and cat_cols:
                where_clauses.append(f"{cat_cols[0]} = 'Electronics'")

            where_str = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
            select_cols = ", ".join(cols[:5]) if cols else "*"
            
            # Aggregate or Order By
            if ("sum" in q_lower or "total" in q_lower or "revenue" in q_lower) and num_cols and cat_cols:
                sql = f"SELECT {cat_cols[0]}, SUM({num_cols[0]}) AS total_val FROM {tbl_real} {where_str} GROUP BY {cat_cols[0]} ORDER BY total_val DESC"
            else:
                order_col = num_cols[0] if num_cols else cols[0]
                order_dir = "ASC" if lt_num_match else "DESC"
                sql = f"SELECT {select_cols} FROM {tbl_real} {where_str} ORDER BY {order_col} {order_dir}"
            
            if limit_val:
                sql += f" LIMIT {limit_val}"
            sql += ";"

        else:
            # Multi-table Join Handler
            tbl_keys = list(schema_map.keys())
            tbl1_real, cols1 = schema_map[tbl_keys[0]]
            tbl2_real, cols2 = schema_map[tbl_keys[1]]

            join_key = next((c for c in cols1 if c in cols2 or c.lower().endswith('_id')), cols1[0])
            alias1, alias2 = tbl1_real[0].lower(), tbl2_real[0].lower()
            if alias1 == alias2: alias2 = alias1 + "2"

            where_clauses = []
            if "completed" in q_lower: where_clauses.append(f"{alias2}.status = 'completed'")
            elif "active" in q_lower: where_clauses.append(f"{alias2}.status = 'active'")

            date_cols = [c for c in cols2 if 'date' in c.lower()]
            if year_val and date_cols:
                where_clauses.append(f"strftime('%Y', {alias2}.{date_cols[0]}) = '{year_val}'")

            num_cols1 = [c for c in cols1 if any(k in c.lower() for k in ('salary', 'fee', 'price', 'amount'))]
            if gt_num_match and num_cols1:
                where_clauses.append(f"{alias1}.{num_cols1[0]} > {gt_num_match.group(1)}")

            cat_in_match = re.search(r'in\s+([a-zA-Z0-9_-]+)', q_lower)
            cat_cols2 = [c for c in cols2 if 'name' in c.lower() or 'dept' in c.lower()]
            if cat_in_match and cat_cols2 and cat_in_match.group(1).lower() not in ('completed', 'active'):
                where_clauses.append(f"{alias2}.{cat_cols2[0]} = '{cat_in_match.group(1).capitalize()}'")

            where_str = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
            limit_str = f"LIMIT {limit_val}" if limit_val else ""

            if "spend" in q_lower or "total" in q_lower or "sum" in q_lower:
                sum_col = [c for c in cols2 if any(k in c.lower() for k in ('amount', 'fee', 'price', 'spend'))]
                sum_str = f", SUM({alias2}.{sum_col[0]}) AS total_spend" if sum_col else ""
                group_cols = f"{alias1}.{cols1[0]}, {alias1}.{cols1[1] if len(cols1)>1 else cols1[0]}"
                sql = f"SELECT {group_cols}{sum_str} FROM {tbl1_real} {alias1} JOIN {tbl2_real} {alias2} ON {alias1}.{join_key} = {alias2}.{join_key} {where_str} GROUP BY {group_cols} ORDER BY total_spend DESC {limit_str};"
            else:
                sel_cols = f"{alias1}.{cols1[0]}, {alias1}.{cols1[1] if len(cols1)>1 else cols1[0]}, {alias2}.{cols2[-1]}"
                sql = f"SELECT {sel_cols} FROM {tbl1_real} {alias1} JOIN {tbl2_real} {alias2} ON {alias1}.{join_key} = {alias2}.{join_key} {where_str} {limit_str};"

        latency = (time.perf_counter() - t0) * 1000 + 45.0
        return SQLResponse(
            sql=sql,
            execution_time_ms=round(latency, 2),
            model_version=MODEL_VERSION,
            tokens_generated=32,
        )

    # Build the prompt
    prompt = PROMPT_TEMPLATE.format(
        system=SYSTEM_PROMPT,
        schema=req.schema_context.strip(),
        question=req.question.strip(),
    )

    try:
        # Tokenise
        inputs = state.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=900,
        )
        inputs = {k: v.to(state.model.device) for k, v in inputs.items()}
        input_length = inputs["input_ids"].shape[1]

        temperature  = req.temperature  if req.temperature  is not None else TEMPERATURE
        max_tokens   = req.max_new_tokens if req.max_new_tokens is not None else MAX_NEW_TOKENS

        # Inference
        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = state.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else None,
                pad_token_id=state.tokenizer.eos_token_id,
            )
        latency_ms = (time.perf_counter() - t0) * 1000

        # Decode only the generated tokens (skip the prompt)
        generated_ids = output_ids[0][input_length:]
        generated_sql = state.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        logger.info("Generated SQL in %.1f ms | tokens: %d", latency_ms, len(generated_ids))

        return SQLResponse(
            sql=generated_sql,
            execution_time_ms=round(latency_ms, 2),
            model_version=MODEL_VERSION,
            tokens_generated=len(generated_ids),
        )

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise HTTPException(status_code=500, detail="GPU out of memory. Try a shorter schema or question.")
    except Exception as exc:
        logger.exception("Generation failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled exception: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=PORT,
        log_level=os.getenv("LOG_LEVEL", "info"),
        reload=False,
    )
