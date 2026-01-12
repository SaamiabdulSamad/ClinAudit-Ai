from langgraph.graph import StateGraph, END, START
from src.workflows.state import AgentState
from src.agents.researcher import researcher_node
from src.agents.auditor import auditor_node
from src.agents.rewriter import rewriter_node
from src.agents.tavily_search import web_search_node
# IMPORT THE ROUTER HERE
from src.workflows.router import routing_logic 

workflow = StateGraph(AgentState)

# ... (add your nodes here)

workflow.add_conditional_edges(
    "auditor",
    routing_logic, # NOW THIS IS DEFINED!
    {
        "finalize": END,
        "retry_local": "rewrite_query",
        "tavily_search": "web_research"
    }
)