import os
from langchain_tavily import TavilySearch
from langchain_core.messages import ToolMessage

def web_search_node(state):
    """
    Executes web research via Tavily.
    Refactored to use 'Lazy Initialization' to prevent boot-time crashes.
    """
    # 1. Verify key ONLY when the node is executed
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        return {"messages": [ToolMessage(content="Error: TAVILY_API_KEY not found", tool_call_id="web_search")]}

    # 2. Initialize the tool locally
    search = TavilySearch(max_results=3)
    
    user_query = state["messages"][-1].content
    results = search.invoke(user_query)
    
    return {"messages": [ToolMessage(content=str(results), tool_call_id="web_search")]}