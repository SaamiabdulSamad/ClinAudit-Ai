# 1. IMMEDIATE ENVIRONMENT CONTROL (MUST BE FIRST)
import os
from dotenv import load_dotenv
load_dotenv() 

import warnings
import time
import hashlib
import json 
import redis
from supabase import create_client, Client

# Suppress noisy tool warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_tavily")

# 2. STANDARD LIBRARY & ASYNC IMPORTS
import uvicorn
import anyio
from pydantic import BaseModel

# 3. THIRD PARTY IMPORTS
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
# 🏥 RESILIENT INFRASTRUCTURE SETUP
# -------------------------
REDIS_URL = os.getenv("REDIS_URL")
redis_client = None
storage_uri = "memory://"  # 👈 Default safe fallback for high availability

if REDIS_URL:
    try:
        # SSL/TLS Handshake Fix for AWS -> Upstash
        if "upstash.io" in REDIS_URL:
            if REDIS_URL.startswith("redis://"):
                REDIS_URL = REDIS_URL.replace("redis://", "rediss://")
            if "ssl_cert_reqs" not in REDIS_URL:
                sep = "&" if "?" in REDIS_URL else "?"
                REDIS_URL += f"{sep}ssl_cert_reqs=none"
        
        # Connection test with 5-second timeout to prevent ASGI hang
        redis_client = redis.from_url(REDIS_URL, decode_responses=True, socket_timeout=5)
        redis_client.ping() 
        storage_uri = REDIS_URL # 👈 Only upgrade to Redis if it's actually alive
        print("✅ PROD: Redis Handshake Successful")
    except Exception as e:
        print(f"⚠️ INFRA WARNING: Redis unreachable ({e}). Using In-Memory Fallback.")

# Supabase Initialization
SB_URL = os.getenv("SUPABASE_URL")
SB_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SB_URL, SB_KEY) if SB_URL and SB_KEY else None

# Rate Limiter - Uses the resilient storage_uri (Redis OR Memory)
limiter = Limiter(key_func=get_remote_address, storage_uri=storage_uri)

# -------------------------
# SECURITY & HELPER LOGIC
# -------------------------
API_KEY_NAME = "X-FactGuard-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key: str = Security(api_key_header)):
    expected_key = os.getenv("FACTGUARD_API_KEY", "dev-secret-123")
    if api_key == expected_key:
        return api_key
    raise HTTPException(status_code=403, detail="Unauthorized: Invalid API Key")

class ClaimRequest(BaseModel):
    claim_text: str

# -------------------------
# FASTAPI CONFIGURATION
# -------------------------
app = FastAPI(title="FactGuard AI - Hardened Production")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.get("/health")
async def health():
    return {
        "status": "healthy", 
        "cache_tier": "redis" if storage_uri != "memory://" else "memory",
        "persistence": "connected" if supabase else "offline"
    }

@app.post("/analyze")
@limiter.limit("5/minute")
async def analyze_claim(request: Request, payload: ClaimRequest, api_key: str = Depends(get_api_key)):
    start_time = time.time()
    claim = payload.claim_text
    
    # 1. SEMANTIC CACHE (Redis) - With silent fail for stability
    if redis_client:
        try:
            cache_key = f"audit:{hashlib.md5(claim.lower().strip().encode()).hexdigest()}"
            cached_data = redis_client.get(cache_key)
            if cached_data:
                return {**json.loads(cached_data), "latency": "cache_hit"}
        except Exception as e:
            print(f"⚠️ Cache connectivity error: {e}")

    try:
        # 2. AGENT EXECUTION
        result = await anyio.to_thread.run_sync(
            agent_graph.invoke, {"messages": [HumanMessage(content=claim)]}
        )
        
        audit_data = result.get("audit_result", {})
        latency_val = time.time() - start_time
        latency_str = f"{latency_val:.2f}s"

        response_body = {
            "claim": claim,
            "audit_score": audit_data.get("faithfulness_score", 0.0),
            "latency": latency_str,
            "reasoning": result["messages"][-1].content if result.get("messages") else "No data.",
            "status": "success"
        }

        # 3. PERMANENT ARCHIVE (Supabase) - Wrapped in exception boundary
        if supabase:
            try:
                supabase.table("medical_audits").insert({
                    "claim_text": claim,
                    "audit_score": response_body["audit_score"],
                    "reasoning": response_body["reasoning"],
                    "latency": latency_str
                }).execute()
            except Exception as e:
                print(f"⚠️ Database logging failure: {e}")

        # 4. UPDATE CACHE (Redis)
        if redis_client:
            try:
                redis_client.setex(cache_key, 3600, json.dumps(response_body))
            except Exception: pass
        
        # Metrics update
        if AGENT_LOOP_LATENCY: AGENT_LOOP_LATENCY.observe(latency_val)

        return response_body

    except Exception as e:
        print(f"🔥 AGENT CRASH: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
# Force Sync 1.0