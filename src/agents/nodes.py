from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from src.schemas.state import AgentState

llm = ChatOpenAI(model="gpt-4o", temperature=0)

def auditor_node(state: AgentState):
    """
    Evaluates the evidence found by the researcher against the patient claim.
    """
    evidence_text = "\n".join(state["evidence"])
    
    prompt = f"""
    You are a Senior Medical Auditor. Your task is to verify a patient's medical claim.
    
    CLAIM: {state['patient_claim']}
    EVIDENCE FOUND: {evidence_text}
    
    Instructions:
    1. If the evidence directly supports the claim according to medical guidelines, respond with 'VERDICT: VALID'.
    2. If the evidence contradicts the claim, respond with 'VERDICT: INVALID'.
    3. If the evidence is missing or too vague, respond with 'VERDICT: RETRY'.
    
    Your response must begin with the verdict.
    """
    
    response = llm.invoke(prompt)
    content = response.content.upper()
    
    if "VALID" in content:
        return {"verdict": "VALID", "messages": [AIMessage(content="Claim verified and approved.")]}
    elif "INVALID" in content:
        return {"verdict": "INVALID", "messages": [AIMessage(content="Claim rejected: Contradicts guidelines.")]}
    else:
        return {"verdict": "RETRY", "messages": [AIMessage(content="Insufficient evidence. Retrying search.")]}