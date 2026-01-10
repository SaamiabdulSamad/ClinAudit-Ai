# 1. IMMEDIATE ENVIRONMENT CONTROL (MUST BE FIRST)
import os
from dotenv import load_dotenv
load_dotenv() # 👈 Load keys BEFORE importing internal agent modules

import warnings
import time
import hashlib
import json
import redis

# Suppress noisy tool warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_tavily")

# 2. STANDARD LIBRARY & ASYNC IMPORTS
import uvicorn
import anyio
from pydantic import BaseModel

# 3. THIRD PARTY IMPORTS
import inngest
import inngest.fast_api
from fastapi import FastAPI, Request, HTTPException, Response, Depends, Security
from fastapi.security import APIKeyHeader
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from langchain_core.messages import HumanMessage
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

# 4. INTERNAL IMPORTS (SAFE TO IMPORT AFTER load_dotenv)
try:
    from src.agents.graph import app as agent_graph
    from src.utils.metrics import AUDIT_FAITHFULNESS_SCORE, HALLUCINATION_COUNT, AGENT_LOOP_LATENCY
except ImportError as e:
    print(f"❌ Initialization Error: Check folder structure or metrics.py! {e}")
    raise

# -------------------------
# Redis & Shared State Setup (THE AWS FOUNDATION)
# -------------------------
RAW_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Architect Guard: Force rediss:// for Upstash/SSL cloud environments
if "upstash.io" in RAW_REDIS_URL and not RAW_REDIS_URL.startswith("rediss://"):
    REDIS_URL = RAW_REDIS_URL.replace("redis://", "rediss://")
else:
    REDIS_URL = RAW_REDIS_URL

# Establish connection with auto-retry logic
redis_client = redis.from_url(REDIS_URL, decode_responses=True, socket_timeout=10)

# -------------------------
# Security & Rate Limiting (THE ARMOR)
# -------------------------
API_KEY_NAME = "X-FactGuard-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key: str = Security(api_key_header)):
    """Enterprise Security: Validates incoming requests."""
    expected_key = os.getenv("FACTGUARD_API_KEY", "dev-secret-123")
    if api_key == expected_key:
        return api_key
    # Brutal Truth: Log the failure for CloudWatch observability
    print(f"❌ AUTH FAILURE: Received '{api_key}'")
    raise HTTPException(status_code=403, detail="Unauthorized: Invalid API Key")

limiter = Limiter(key_func=get_remote_address, storage_uri=REDIS_URL)

# -------------------------
# Data Models (THE BLUEPRINT)
# -------------------------
class ClaimRequest(BaseModel):
    """Defines the professional JSON structure for incoming claims."""
    claim_text: str

# -------------------------
# Semantic Caching Logic
# -------------------------
def get_cached_audit(claim: str):
    cache_key = f"audit:{hashlib.md5(claim.lower().strip().encode()).hexdigest()}"
    try:
        cached_data = redis_client.get(cache_key)
        return json.loads(cached_data) if cached_data else None
    except Exception as e:
        print(f"⚠️ Redis Read Error: {e}")
        return None

def set_cached_audit(claim: str, response: dict):
    cache_key = f"audit:{hashlib.md5(claim.lower().strip().encode()).hexdigest()}"
    try:
        # Cache for 1 hour to reduce token costs
        redis_client.setex(cache_key, 3600, json.dumps(response))
    except Exception as e:
        print(f"⚠️ Redis Write Error: {e}")

# -------------------------
# Business Logic
# -------------------------
def get_business_decision(state: dict):
    audit = state.get("audit_result", {})
    score = audit.get("faithfulness_score", 0.0)
    verdict = audit.get("verdict", "FAIL")
    
    if score == 0.0 and verdict == "FAIL":
        return "DENIED (Policy Explicitly Excludes This)", 1.0
    if verdict == "PASS" and score >= 0.85:
        return "APPROVED", score
    return "NEEDS_HUMAN_REVIEW (Insufficient Groundedness)", score

# -------------------------
# Inngest Setup
# -------------------------
inngest_client = inngest.Inngest(app_id="factguard_app", is_production=False)

@inngest_client.create_function(
    fn_id="medical-audit-workflow",
    trigger=inngest.TriggerEvent(event="audit/claim.submitted"),
)
async def audit_workflow(ctx):
    claim_text = ctx.event.data.get("patient_claim")
    if not claim_text: return {"status": "failed"}
    
    result = await anyio.to_thread.run_sync(
        agent_graph.invoke, {"messages": [HumanMessage(content=claim_text)]}
    )
    decision, conf = get_business_decision(result)
    return {"status": "completed", "decision": decision}

# -------------------------
# FastAPI App
# -------------------------
app = FastAPI(title="FactGuard AI - Medical Claim Auditor")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
async def health_check():
    try:
        redis_client.ping()
        return {"status": "healthy", "redis": "connected"}
    except Exception as e:
        return {"status": "degraded", "redis": f"disconnected: {str(e)}"}

@app.post("/analyze")
@limiter.limit("10/minute")
async def analyze_claim(
    request: Request, 
    payload: ClaimRequest, 
    api_key: str = Depends(get_api_key)
):
    start_time = time.time()
    claim = payload.claim_text
    
    # 1. SHARED CACHE CHECK
    cached_result = get_cached_audit(claim)
    if cached_result:
        return {**cached_result, "latency": "cache_hit"}

    try:
        # 2. AGENT EXECUTION
        result = await anyio.to_thread.run_sync(
            agent_graph.invoke, {"messages": [HumanMessage(content=claim)]}
        )
        
        audit_data = result.get("audit_result", {})
        score = audit_data.get("faithfulness_score", 0.0)
        verdict = audit_data.get("verdict", "FAIL")
        
        latency = time.time() - start_time
        
        # Log metrics for Prometheus
        AGENT_LOOP_LATENCY.observe(latency)
        AUDIT_FAITHFULNESS_SCORE.labels(claim_type="medical").set(score)
        
        if verdict == "PASS" and score < 0.6:
            HALLUCINATION_COUNT.inc()

        decision, confidence = get_business_decision(result)

        response_body = {
            "claim": claim,
            "decision": decision,
            "confidence": confidence,
            "audit_score": score,
            "latency": f"{latency:.2f}s",
            "reasoning": result["messages"][-1].content if result.get("messages") else "No reasoning available.",
            "status": "success"
        }

        # 3. UPDATE SHARED CACHE
        set_cached_audit(claim, response_body)

        return response_body
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agentic Execution Failed: {str(e)}")

# -------------------------
# STARTUP
# -------------------------
try:
    inngest.fast_api.serve(app, inngest_client, [audit_workflow])
    print("✅ Inngest routes registered successfully")
except Exception as e:
    print(f"❌ Inngest Registration Failed: {e}")

if __name__ == "__main__":
    # Standard production port
    uvicorn.run("src.main:app", host="0.0.0.0", port=8001, reload=True)