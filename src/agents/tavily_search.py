from langchain_tavily import TavilySearch
from langchain_core.messages import ToolMessage
from dotenv import load_dotenv

load_dotenv()

# Initialize the search tool using the correct new class name
search = TavilySearch(max_results=3)

def web_search_node(state):
    # This node triggers when the Auditor needs web data
    user_query = state["messages"][-1].content
    results = search.invoke(user_query)
    
    # We return a ToolMessage so the agent knows this is an external result
    return {"messages": [ToolMessage(content=str(results), tool_call_id="web_search")]}