import os
import sys

# Standardize pathing for AWS App Runner
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# CRITICAL: Removed 'src.' to match container PYTHONPATH
from agents.graph import app as agent_graph 
from workflows.state import AgentState

app = FastAPI(title="ClinAudit AI API")

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """AWS App Runner Health Check Endpoint"""
    return {
        "status": "healthy", 
        "version": "1.2.0",
        "database_connected": os.getenv("QDRANT_URL") is not None
    }

@app.get("/")
async def root():
    return {"message": "ClinAudit AI Agent is Live"}

@app.post("/analyze")
async def analyze_claim(request: dict):
    """Entry point for the AI Auditor"""
    try:
        # Initialize state
        initial_state = {
            "messages": [("user", request.get("claim_text", ""))],
            "retry_count": 0
        }
        # Run the graph
        result = await agent_graph.ainvoke(initial_state)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)