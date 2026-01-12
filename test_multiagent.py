from src.agents.graph import app as agent_graph
from langchain_core.messages import HumanMessage

print("\n" + "="*70)
print("TESTING MULTI-AGENT SELF-CORRECTION")
print("="*70)

result = agent_graph.invoke({
    "messages": [HumanMessage(content="Does aspirin prevent heart attacks?")],
    "retry_count": 0,
    "needs_web_search": False
})

print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)
print(f"Messages exchanged: {len(result['messages'])}")
print(f"Final verdict: {result['messages'][-1].content}")
print(f"Faithfulness score: {result.get('audit_result', {}).get('faithfulness_score', 'N/A')}")
print(f"Retry count: {result.get('retry_count', 0)}")
print(f"Needed web search: {result.get('needs_web_search', False)}")
print("="*70)