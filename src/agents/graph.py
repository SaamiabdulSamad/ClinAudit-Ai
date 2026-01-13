import sys
import os

# 1. ROOT PATH FIX: Ensures 'agents' and 'workflows' are findable
# This handles the "No module named src" error by looking at the project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from langgraph.graph import StateGraph, END, START
from workflows.state import AgentState
from agents.router import routing_logic
from agents.researcher import researcher_node
from agents.auditor import auditor_node
from agents.tavily_search import web_search_node

from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings

# -------------------------
# 2. Resilient Knowledge Base Initialization
# -------------------------
def get_vectorstore():
    """
    DECOUPLED LOADER: Prevents the app from crashing during boot.
    If Qdrant is down, the app stays alive so you can check logs.
    """
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_KEY")
    
    # Validation check to catch empty variables before they crash the client
    if not url or "qdrant.io" not in url:
        print("⚠️ QDRANT_URL is missing or malformed in AWS Environment.")
        return None

    try:
        # Port 6333 is required for Qdrant Cloud Python Client
        vs = QdrantVectorStore.from_existing_collection(
            embedding=OpenAIEmbeddings(),
            collection_name="pubmed_docs", 
            url=url,
            api_key=api_key,
        )
        print("✅ PROD: Qdrant Handshake Successful")
        return vs
    except Exception as e:
        # This print will finally show up in your "Application Logs"
        print(f"❌ PROD ERROR: Qdrant Connection Failed -> {e}")
        return None

# Initialize as a global variable but safely handled via the function
vectorstore = get_vectorstore()

# -------------------------
# 3. Initialize the State Machine
# -------------------------
workflow = StateGraph(AgentState)

workflow.add_node("local_research", researcher_node) 
workflow.add_node("auditor", auditor_node)
workflow.add_node("web_research", web_search_node)
workflow.add_node("increment_retry", lambda state: {"retry_count": state.get("retry_count", 0) + 1})

workflow.add_edge(START, "local_research")
workflow.add_edge("local_research", "auditor")

workflow.add_conditional_edges(
    "auditor",
    routing_logic,
    {
        "retry_research": "increment_retry",
        "tavily_search": "web_research",
        "finalize": END
    }
)

workflow.add_edge("increment_retry", "local_research")
workflow.add_edge("web_research", "auditor")

app = workflow.compile()