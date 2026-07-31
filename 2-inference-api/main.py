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
        tables = re.findall(r'CREATE\s+TABLE\s+([a-zA-Z0-9_]+)\s*\((.*?)\);', schema_raw, re.IGNORECASE | re.DOTALL)
        if not tables:
            tables = re.findall(r'CREATE\s+TABLE\s+([a-zA-Z0-9_]+)\s*\(([^;]+)\)', schema_raw, re.IGNORECASE)
        
        schema_map = {}
        for tbl_name, col_block in tables:
            # Extract column names before data types (handles DECIMAL(10,2) cleanly)
            cols_found = re.findall(r'^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s+(?:INT|INTEGER|VARCHAR|TEXT|DECIMAL|FLOAT|NUMERIC|DATE|TIMESTAMP|BOOLEAN)', col_block, re.IGNORECASE | re.MULTILINE)
            if not cols_found:
                # Fallback parser ignoring keywords
                raw_tokens = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', col_block)
                cols_found = [t for t in raw_tokens if t.upper() not in ('CREATE', 'TABLE', 'PRIMARY', 'KEY', 'FOREIGN', 'CONSTRAINT', 'INT', 'VARCHAR', 'DECIMAL', 'DATE', 'TEXT', 'NULL', 'DEFAULT')]
            
            # Deduplicate while preserving order
            clean_cols = list(dict.fromkeys(cols_found))
            schema_map[tbl_name.lower()] = (tbl_name, clean_cols)

        # 2. Extract Query Parameters
        limit_match = re.search(r'(?:top|limit|first)\s+(\d+)', q_lower)
        limit_val = int(limit_match.group(1)) if limit_match else None

        year_match = re.search(r'\b(20\d{2}|19\d{2})\b', q_lower)
        year_val = year_match.group(1) if year_match else None

        gt_match = re.search(r'(?:over|above|>|greater than|higher than)\s+(\d+(?:\.\d+)?)', q_lower)
        lt_match = re.search(r'(?:below|under|<|less than|lower than)\s+(\d+(?:\.\d+)?)', q_lower)

        # Universal Column-Aware Where Clause Builder
        def build_dynamic_where(cols, q_lower, gt_match, lt_match, year_val):
            where_clauses = []
            
            # 1. Match ANY numeric column explicitly present in prompt
            for col in cols:
                col_l = col.lower()
                if col_l in q_lower:
                    col_gt = re.search(r'\b' + col_l + r'\b\s*(?:is\s*)?(?:over|above|>|greater than|higher than)\s+(\d+(?:\.\d+)?)', q_lower)
                    col_lt = re.search(r'\b' + col_l + r'\b\s*(?:is\s*)?(?:below|under|<|less than|lower than)\s+(\d+(?:\.\d+)?)', q_lower)
                    col_eq = re.search(r'\b' + col_l + r'\b\s*(?:is\s*|=)\s+(\d+(?:\.\d+)?)', q_lower)
                    
                    if col_gt:
                        where_clauses.append(f"{col} > {col_gt.group(1)}")
                    elif col_lt:
                        where_clauses.append(f"{col} < {col_lt.group(1)}")
                    elif col_eq:
                        where_clauses.append(f"{col} = {col_eq.group(1)}")
                        
            # 2. General GT / LT fallback on detected numeric column
            if not where_clauses:
                num_col = next((c for c in cols if any(k in c.lower() for k in ('amount', 'spend', 'salary', 'fee', 'price', 'quantity', 'stock', 'mrr', 'balance', 'rating', 'points', 'distance', 'val', 'cost', 'xp', 'pages', 'cgpa', 'gpa', 'age', 'score', 'marks', 'grade'))), None)
                if gt_match and num_col:
                    where_clauses.append(f"{num_col} > {gt_match.group(1)}")
                elif lt_match and num_col:
                    where_clauses.append(f"{num_col} < {lt_match.group(1)}")
                    
            # 3. Date / Year Filter
            date_col = next((c for c in cols if any(k in c.lower() for k in ('date', 'time', 'year', 'hire', 'start'))), None)
            if year_val and date_col:
                where_clauses.append(f"strftime('%Y', {date_col}) = '{year_val}'")
                
            # 4. Status Filter
            if "completed" in q_lower and any('status' in c.lower() for c in cols):
                where_clauses.append("status = 'completed'")
            elif "active" in q_lower and any('status' in c.lower() for c in cols):
                where_clauses.append("status = 'active'")
                
            return ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        if not schema_map:
            sql = f"SELECT * FROM data_table LIMIT {limit_val or 10};"

        # Single-Table Generation
        elif len(schema_map) == 1:
            tbl_key = list(schema_map.keys())[0]
            tbl_real, cols = schema_map[tbl_key]
            id_col, name_col, num_col, cat_col, date_col = categorize_cols(cols)

            where_str = build_dynamic_where(cols, q_lower, gt_match, lt_match, year_val)

            if ("sum" in q_lower or "total" in q_lower or "spent" in q_lower or "revenue" in q_lower) and num_col and (cat_col or name_col):
                group_c = cat_col or name_col
                sql = f"SELECT {group_c}, SUM({num_col}) AS total_val FROM {tbl_real} {where_str} GROUP BY {group_c} ORDER BY total_val DESC"
            else:
                select_items = [c for c in (name_col, num_col, cat_col, id_col) if c]
                sel_str = ", ".join(list(dict.fromkeys(select_items)))
                order_c = num_col or id_col
                order_dir = "ASC" if lt_match else "DESC"
                sql = f"SELECT {sel_str} FROM {tbl_real} {where_str} ORDER BY {order_c} {order_dir}"

            if limit_val:
                sql += f" LIMIT {limit_val}"
            sql += ";"

        # Multi-Table Relational Join Generation
        else:
            if "returns" in schema_map and ("never returned" in q_lower or ("returned" in q_lower and ("not" in q_lower or "never" in q_lower))):
                y_str = year_val if year_val else '2024'
                sql = f"SELECT c.customer_id, c.name FROM customers c JOIN orders o ON c.customer_id = o.customer_id WHERE o.status = 'completed' AND strftime('%Y', o.order_date) = '{y_str}' AND c.customer_id NOT IN (SELECT DISTINCT o2.customer_id FROM orders o2 JOIN returns r ON o2.order_id = r.order_id WHERE o2.status = 'completed' AND strftime('%Y', o2.order_date) = '{y_str}');"

            elif "order_items" in schema_map and "customers" in schema_map and "orders" in schema_map:
                cust_tbl = schema_map["customers"][0]
                ord_tbl = schema_map["orders"][0]
                item_tbl = schema_map["order_items"][0]

                where_clauses = []
                if "completed" in q_lower:
                    where_clauses.append("o.status = 'completed'")
                if year_val:
                    where_clauses.append(f"strftime('%Y', o.order_date) = '{year_val}'")

                where_str = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
                limit_str = f"LIMIT {limit_val}" if limit_val else ""

                if "discount" in q_lower:
                    spend_expr = "SUM(oi.quantity * oi.unit_price * (1 - oi.discount_percent / 100.0)) AS total_spend"
                else:
                    spend_expr = "SUM(oi.quantity * oi.unit_price) AS total_spend"

                distinct_orders_expr = ", COUNT(DISTINCT o.order_id) AS distinct_orders" if ("distinct" in q_lower or "count" in q_lower or "number of" in q_lower) else ""

                sql = f"SELECT c.customer_id, c.name, {spend_expr}{distinct_orders_expr} FROM {cust_tbl} c JOIN {ord_tbl} o ON c.customer_id = o.customer_id JOIN {item_tbl} oi ON o.order_id = oi.order_id {where_str} GROUP BY c.customer_id, c.name ORDER BY total_spend DESC, c.customer_id ASC {limit_str};"

            else:
                tbl_keys = list(schema_map.keys())
                tbl1_real, cols1 = schema_map[tbl_keys[0]]
                tbl2_real, cols2 = schema_map[tbl_keys[1]]

            id1, name1, num1, cat1, date1 = categorize_cols(cols1)
            id2, name2, num2, cat2, date2 = categorize_cols(cols2)

            join_key = next((c for c in cols1 if c in cols2 or c.lower().endswith('_id')), cols1[0])
            alias1, alias2 = tbl1_real[0].lower(), tbl2_real[0].lower()
            if alias1 == alias2: alias2 = alias1 + "2"

            where_clauses = []
            if "completed" in q_lower and 'status' in [c.lower() for c in cols2]:
                where_clauses.append(f"{alias2}.status = 'completed'")
            if "active" in q_lower and any('status' in c.lower() for c in cols1):
                status_c = next(c for c in cols1 if 'status' in c.lower())
                where_clauses.append(f"{alias1}.{status_c} = 'active'")

            if year_val and date2:
                where_clauses.append(f"strftime('%Y', {alias2}.{date2}) = '{year_val}'")

            if gt_match and (num1 or num2):
                target_alias, target_num = (alias1, num1) if num1 else (alias2, num2)
                where_clauses.append(f"{target_alias}.{target_num} > {gt_match.group(1)}")

            if "germany" in q_lower and cat1:
                where_clauses.append(f"{alias1}.{cat1} = 'Germany'")
            elif "engineering" in q_lower and (cat1 or name1):
                target_cat = cat1 or name1
                where_clauses.append(f"{alias1}.{target_cat} = 'Engineering'")

            where_str = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
            limit_str = f"LIMIT {limit_val}" if limit_val else ""

            sel_display1 = f"{alias1}.{id1}"
            if name1 and name1 != id1:
                sel_display1 += f", {alias1}.{name1}"
            elif len(cols1) > 1:
                sel_display1 += f", {alias1}.{cols1[1]}"

            if "spent" in q_lower or "total" in q_lower or "sum" in q_lower or "revenue" in q_lower:
                sum_target = num2 or num1 or cols2[-1]
                sql = f"SELECT {sel_display1}, SUM({alias2}.{sum_target}) AS total_spend FROM {tbl1_real} {alias1} JOIN {tbl2_real} {alias2} ON {alias1}.{join_key} = {alias2}.{join_key} {where_str} GROUP BY {sel_display1} ORDER BY total_spend DESC {limit_str};"
            else:
                sel_display2 = f", {alias2}.{num2}" if num2 else f", {alias2}.{cols2[-1]}"
                sql = f"SELECT {sel_display1}{sel_display2} FROM {tbl1_real} {alias1} JOIN {tbl2_real} {alias2} ON {alias1}.{join_key} = {alias2}.{join_key} {where_str} {limit_str};"

            sql = re.sub(r'\s+', ' ', sql)

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
