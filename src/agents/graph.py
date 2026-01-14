import sys
import os

# ROOT PATH FIX: Ensures sibling directories like 'workflows' are accessible
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
    Prevents Exit Code 3. If the connection fails, the app still boots
    allowing you to debug via logs instead of the container just dying.
    """
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_KEY")
    
    if not url or "qdrant.io" not in url:
        print("⚠️ QDRANT_URL is not configured correctly.")
        return None

    try:
        # Initializing here ensures port 6333 is respected for Cloud
        vs = QdrantVectorStore.from_existing_collection(
            embedding=OpenAIEmbeddings(),
            collection_name="pubmed_docs", 
            url=url,
            api_key=api_key,
        )
        print("SUCCESS: Qdrant Handshake Completed")
        return vs
    except Exception as e:
        print(f"❌ CONNECTION FAILED: Qdrant is unreachable: {e}")
        return None

# Global instance for nodes to utilize
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

# Build the flow logic
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