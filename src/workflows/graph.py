import sys
import os

# ROOT PATH FIX: Ensures internal modules are findable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langgraph.graph import StateGraph, END, START
from workflows.state import AgentState
from agents.router import routing_logic
from agents.researcher import researcher_node
from agents.auditor import auditor_node
from agents.tavily_search import web_search_node

from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings

# -------------------------
# 1. Resilient Knowledge Base Initialization
# -------------------------
def get_vectorstore():
    """
    Lazy loader to prevent 'Exit Code 3'. 
    This prevents a crash if Qdrant is slow to respond.
    """
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_KEY")
    
    if not url or "qdrant.io" not in url:
        print("⚠️ WARNING: QDRANT_URL environment variable is missing.")
        return None

    try:
        # Initializing here prevents 'ImportTime' connection errors
        vs = QdrantVectorStore.from_existing_collection(
            embedding=OpenAIEmbeddings(),
            collection_name="pubmed_docs", 
            url=url,
            api_key=api_key,
        )
        print("✅ SUCCESS: Qdrant Handshake Completed")
        return vs
    except Exception as e:
        print(f"❌ ERROR: Qdrant Connection Failed: {e}")
        return None

# Global instance for the graph nodes to use
vectorstore = get_vectorstore()

# -------------------------
# 2. Initialize the State Machine
# -------------------------
workflow = StateGraph(AgentState)

# Define nodes
workflow.add_node("local_research", researcher_node) 
workflow.add_node("auditor", auditor_node)
workflow.add_node("web_research", web_search_node)
workflow.add_node("increment_retry", lambda state: {"retry_count": state.get("retry_count", 0) + 1})

# Build the flow
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