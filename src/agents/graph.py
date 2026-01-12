import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from langgraph.graph import StateGraph, END, START
from workflows.state import AgentState
from src.agents.router import routing_logic
from src.agents.researcher import researcher_node
from src.agents.auditor import auditor_node
from src.agents.tavily_search import web_search_node

# 📚 NEW: PubMed Integration Imports
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import create_retriever_tool
# -------------------------
# 1. Initialize the Specialized Knowledge Base
# -------------------------
# We initialize this globally so the graph doesn't re-connect on every call
vectorstore = QdrantVectorStore.from_existing_collection(
    embedding=OpenAIEmbeddings(),
    collection_name="pubmed_docs", # 👈 Ensure this matches your Qdrant collection name
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_KEY"),
)

pubmed_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# -------------------------
# 2. Graph Support Nodes
# -------------------------
def increment_retry(state: AgentState) -> AgentState:
    """Helper node to increment retry counter before looping back"""
    return {
        "retry_count": state.get("retry_count", 0) + 1
    }

# -------------------------
# 3. Initialize the State Machine
# -------------------------
workflow = StateGraph(AgentState)

# Define nodes
# Note: Ensure researcher_node in src/agents/researcher.py is updated 
# to use the pubmed_retriever tool!
workflow.add_node("local_research", researcher_node) 
workflow.add_node("auditor", auditor_node)
workflow.add_node("web_research", web_search_node)
workflow.add_node("increment_retry", increment_retry)

# Build the flow
workflow.add_edge(START, "local_research")
workflow.add_edge("local_research", "auditor")

# CRITICAL: Conditional routing from auditor
workflow.add_conditional_edges(
    "auditor",
    routing_logic,
    {
        "retry_research": "increment_retry",  # Loop back if local data was thin
        "tavily_search": "web_research",      # Fallback to general internet
        "finalize": END                        # Accept and exit
    }
)

# After incrementing retry, go back to researcher
workflow.add_edge("increment_retry", "local_research")

# After web search, audit again to verify internet findings
workflow.add_edge("web_research", "auditor")

app = workflow.compile()