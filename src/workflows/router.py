# src/workflows/router.py
from src.workflows.state import AgentState

def routing_logic(state: AgentState):
    """
    The Brain of the Graph: Decides the next hop based on audit quality.
    """
    audit_result = state.get("audit_result", {})
    # Use .get() with defaults to prevent the graph from crashing on None values
    score = audit_result.get("faithfulness_score", 0.0)
    retry_count = state.get("retry_count", 0)

    print(f"\n🚦 ROUTER: Current Score: {score} | Retries: {retry_count}")

    # OPTION 1: Success (Pass or clear Fail)
    # If the score is high, it means the Auditor is CONFIDENT in its finding.
    if score >= 0.8:
        print("   Goal Met: Auditor is confident. Finalizing...")
        return "finalize"

    # OPTION 2: Try Local Again (The Corrective Loop)
    # If the score is low but we haven't tried a 'refined' search yet.
    if retry_count < 1:
        print("   🔄 Low Confidence: Triggering Local Rewriter for a second pass...")
        return "retry_local"

    # OPTION 3: Fallback to Web
    # Only spend money/latency on Tavily if the PDF definitely doesn't have it.
    print("   🌐 Local Data Insufficient after retry: Escalating to Web Search...")
    return "tavily_search"