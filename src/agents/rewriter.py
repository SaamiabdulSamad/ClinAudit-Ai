# src/agents/rewriter.py
from langchain_core.messages import HumanMessage
from src.workflows.state import AgentState # <--- THE FIX

def rewriter_node(state: AgentState):
    """
    REWRITER: Takes the failed audit and improves the search query.
    """
    print("🔄 REWRITER: Refining query based on failed audit...")
    user_query = state["messages"][0].content
    
    # We add specific keywords to target the PDF tables we missed
    refined_query = f"{user_query} policy exclusions and non-covered criteria"
    
    return {
        "messages": [HumanMessage(content=refined_query)],
        "retry_count": state.get("retry_count", 0) + 1
    }