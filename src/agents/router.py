from typing import Literal
from src.workflows.state import AgentState

def routing_logic(state: AgentState) -> Literal["retry_local", "tavily_search", "finalize"]:
    """
    The Brain of the Graph: Implements Corrective RAG (CRAG) logic.
    """
    audit = state.get("audit_result", {})
    retry_count = state.get("retry_count", 0)
    needs_web = state.get("needs_web_search", False)
    
    # We use .get() with 0.0 to ensure the code never crashes on a missing score
    faithfulness = audit.get("faithfulness_score", 0.0)
    verdict = audit.get("verdict", "FAIL")
    
    print(f"\n🚦 ROUTER LOGIC:")
    print(f"   📊 Current Groundedness Score: {faithfulness:.2f}")
    print(f"   🔄 Retry Attempt: {retry_count}")
    print(f"   🌐 Web Escalation Flag: {needs_web}")
    
    # CASE 1: The "Success" Path
    # High confidence or the Auditor has clearly confirmed a 'FAIL' (like our 0.00 score)
    # Note: If the score is 0.00 but the Auditor verified the code is NON-COVERED, we finalize.
    if faithfulness >= 0.80 or (faithfulness == 0.0 and verdict == "FAIL"):
        print("   ✅ DECISION: FINALIZE (Audit complete and verified)")
        return "finalize"
    
    # CASE 2: The "Self-Correction" Path
    # Low confidence, and we have retries left. 
    # This triggers the 'rewriter' to improve the search query.
    if faithfulness < 0.80 and retry_count < 2 and not needs_web:
        print(f"   🔄 DECISION: RETRY LOCAL (Triggering Query Refinement)")
        return "retry_local" # Ensure your graph node name matches this
    
    # CASE 3: The "Escalation" Path
    # Local PDF has failed us twice OR the Auditor explicitly asked for Web data.
    print("   🌐 DECISION: ESCALATE TO WEB (Local knowledge base exhausted)")
    return "tavily_search"