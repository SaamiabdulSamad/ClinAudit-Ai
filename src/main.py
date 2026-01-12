# 1. IMMEDIATE ENVIRONMENT CONTROL (MUST BE FIRST)
import os
from dotenv import load_dotenv
load_dotenv() 

import warnings
import time
import hashlib
import json 
import redis  # 👈 RESTORED
from supabase import create_client, Client # 👈 NEW

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

# 4. INTERNAL IMPORTS
try:
    from src.agents.graph import app as agent_graph
    from src.utils.metrics import AUDIT_FAITHFULNESS_SCORE, HALLUCINATION_COUNT, AGENT_LOOP_LATENCY
except ImportError as e:
    print(f"❌ Initialization Error: {e}")
    raise

# -------------------------
# Redis & Supabase Setup
# -------------------------
# SSL FIX: Added ?ssl_cert_reqs=none for cloud environment stability
REDIS_URL = os.getenv("REDIS_URL")
redis_client = None
if REDIS_URL:
    # Append SSL fix if missing
    if "upstash.io" in REDIS_URL and "ssl_cert_reqs" not in REDIS_URL:
        REDIS_URL += "?ssl_cert_reqs=none"
    redis_client = redis.from_url(REDIS_URL, decode_responses=True, socket_timeout=10)

# Supabase Initialization
SB_URL = os.getenv("SUPABASE_URL")
SB_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SB_URL, SB_KEY) if SB_URL and SB_KEY else None

# Rate Limiter
limiter = Limiter(key_func=get_remote_address, storage_uri=REDIS_URL)

# -------------------------
# Security & Helper Logic
# -------------------------
API_KEY_NAME = "X-FactGuard-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key: str = Security(api_key_header)):
    expected_key = os.getenv("FACTGUARD_API_KEY", "dev-secret-123")
    if api_key == expected_key:
        return api_key
    raise HTTPException(status_code=403, detail="Unauthorized: Invalid API Key")

def get_business_decision(state: dict):
    audit = state.get("audit_result", {})
    score = audit.get("faithfulness_score", 0.0)
    verdict = audit.get("verdict", "FAIL")
    if verdict == "PASS" and score >= 0.85:
        return "APPROVED", score
    return "NEEDS_HUMAN_REVIEW", score

class ClaimRequest(BaseModel):
    claim_text: str

# -------------------------
# FastAPI App Configuration
# -------------------------
app = FastAPI(title="FactGuard AI - Hardened Production")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/analyze")
@limiter.limit("5/minute")  # 👈 RATE LIMIT RE-ENABLED
async def analyze_claim(request: Request, payload: ClaimRequest, api_key: str = Depends(get_api_key)):
    start_time = time.time()
    claim = payload.claim_text
    
    # 1. SEMANTIC CACHE (Redis)
    if redis_client:
        cache_key = f"audit:{hashlib.md5(claim.lower().strip().encode()).hexdigest()}"
        try:
            cached_data = redis_client.get(cache_key)
            if cached_data:
                return {**json.loads(cached_data), "latency": "cache_hit"}
        except Exception as e:
            print(f"⚠️ Redis Error: {e}")

    try:
        # 2. AGENT EXECUTION
        result = await anyio.to_thread.run_sync(
            agent_graph.invoke, {"messages": [HumanMessage(content=claim)]}
        )
        
        audit_data = result.get("audit_result", {})
        decision, confidence = get_business_decision(result)
        latency_val = time.time() - start_time
        latency_str = f"{latency_val:.2f}s"

        response_body = {
            "claim": claim,
            "decision": decision,
            "confidence": confidence,
            "audit_score": audit_data.get("faithfulness_score", 0.0),
            "latency": latency_str,
            "reasoning": result["messages"][-1].content if result.get("messages") else "No reasoning.",
            "status": "success"
        }

        # 3. PERMANENT ARCHIVE (Supabase)
        if supabase:
            try:
                supabase.table("medical_audits").insert({
                    "claim_text": claim,
                    "decision": decision,
                    "confidence": confidence,
                    "audit_score": audit_data.get("faithfulness_score", 0.0),
                    "reasoning": response_body["reasoning"],
                    "latency": latency_str
                }).execute()
            except Exception as e:
                print(f"⚠️ Supabase Error: {e}")

        # 4. UPDATE CACHE (Redis)
        if redis_client:
            try:
                redis_client.setex(cache_key, 3600, json.dumps(response_body))
            except Exception as e:
                print(f"⚠️ Cache Update Error: {e}")
        
        # Metrics
        if AGENT_LOOP_LATENCY: AGENT_LOOP_LATENCY.observe(latency_val)

        return response_body

    except Exception as e:
        print(f"🔥 AGENT CRASH: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Execution Failed: {str(e)}")

# Metrics endpoint for Prometheus
@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)