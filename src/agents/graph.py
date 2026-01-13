import sys
import os
# Fix the pathing so it works in both local and AWS environments
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
# 1. Resilient Knowledge Base Initialization
# -------------------------
def get_vectorstore():
    """
    Lazy loader for Qdrant. This prevents the 'Exit Code 3' crash 
    if the network is slow during container boot.
    """
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_KEY")
    
    if not url or "qdrant.io" not in url:
        print("⚠️ QDRANT_URL is missing or malformed. Check AWS Variables.")
        return None

    try:
        vs = QdrantVectorStore.from_existing_collection(
            embedding=OpenAIEmbeddings(),
            collection_name="pubmed_docs", 
            url=url,
            api_key=api_key,
        )
        print("✅ Qdrant Connection Established")
        return vs
    except Exception as e:
        print(f"❌ Qdrant Connection Failed: {e}")
        return None

# Initialize outside of the nodes but wrapped in a check
vectorstore = get_vectorstore()

# -------------------------
# 2. Graph Support Nodes
# -------------------------
def increment_retry(state: AgentState) -> AgentState:
    return {"retry_count": state.get("retry_count", 0) + 1}

# -------------------------
# 3. Initialize the State Machine
# -------------------------
workflow = StateGraph(AgentState)

workflow.add_node("local_research", researcher_node) 
workflow.add_node("auditor", auditor_node)
workflow.add_node("web_research", web_search_node)
workflow.add_node("increment_retry", increment_retry)

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