# 1. IMMEDIATE ENVIRONMENT CONTROL (MUST BE FIRST)
import os
from dotenv import load_dotenv
load_dotenv() 

import warnings
import time
import hashlib
import json 
# import redis # 👈 REMOVED to prevent connection attempts

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
# from slowapi import Limiter, _rate_limit_exceeded_handler # 👈 REMOVED
# from slowapi.util import get_remote_address
# from slowapi.errors import RateLimitExceeded
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
# Security Setup
# -------------------------
API_KEY_NAME = "X-FactGuard-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key: str = Security(api_key_header)):
    expected_key = os.getenv("FACTGUARD_API_KEY", "dev-secret-123")
    if api_key == expected_key:
        return api_key
    print(f"❌ AUTH FAILURE: Received '{api_key}'")
    raise HTTPException(status_code=403, detail="Unauthorized: Invalid API Key")

# -------------------------
# Data Models
# -------------------------
class ClaimRequest(BaseModel):
    claim_text: str

# -------------------------
# Business Decisions
# -------------------------
def get_business_decision(state: dict):
    audit = state.get("audit_result")
    if not audit or not isinstance(audit, dict):
        return "NEEDS_HUMAN_REVIEW (Agent Failed)", 0.0
        
    score = audit.get("faithfulness_score", 0.0)
    verdict = audit.get("verdict", "FAIL")
    
    if score == 0.0 and verdict == "FAIL":
        return "DENIED (Policy Explicitly Excludes This)", 1.0
    if verdict == "PASS" and score >= 0.85:
        return "APPROVED", score
    return "NEEDS_HUMAN_REVIEW (Insufficient Groundedness)", score

# -------------------------
# FastAPI App
# -------------------------
app = FastAPI(title="FactGuard AI - Medical Claim Auditor")
# app.state.limiter = limiter # 👈 DISABLED

@app.post("/analyze")
# @limiter.limit("10/minute") # 👈 REMOVED: Bypassing broken Redis auth
async def analyze_claim(request: Request, payload: ClaimRequest, api_key: str = Depends(get_api_key)):
    start_time = time.time()
    claim = payload.claim_text
    
    # 1. CACHE CHECK DISABLED (Bypassing Redis)
    # cached_result = get_cached_audit(claim)

    try:
        # 2. AGENT EXECUTION
        result = await anyio.to_thread.run_sync(
            agent_graph.invoke, {"messages": [HumanMessage(content=claim)]}
        )
        
        # 3. DEFENSIVE DATA EXTRACTION
        audit_data = result.get("audit_result", {})
        if not audit_data:
             raise ValueError("Agent graph returned empty state.")

        decision, confidence = get_business_decision(result)
        latency = time.time() - start_time
        
        # Metrics
        if AGENT_LOOP_LATENCY: AGENT_LOOP_LATENCY.observe(latency)

        response_body = {
            "claim": claim,
            "decision": decision,
            "confidence": confidence,
            "audit_score": audit_data.get("faithfulness_score", 0.0),
            "latency": f"{latency:.2f}s",
            "reasoning": result["messages"][-1].content if result.get("messages") else "No reasoning.",
            "status": "success"
        }

        # 4. CACHE UPDATE DISABLED
        return response_body

    except Exception as e:
        print(f"🔥 AGENT CRASH: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Execution Failed: {str(e)}")

# ... (Remaining Inngest setup stays same)