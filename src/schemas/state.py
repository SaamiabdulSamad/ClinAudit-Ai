from typing import TypedDict, Annotated, List, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    patient_claim: str
    evidence: List[str]
    verdict: str  # New: "VALID", "INVALID", or "INSUFFICIENT_DATA"
    retry_count: int