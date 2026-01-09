# src/workflows/state.py
from typing import TypedDict, Annotated, List, Dict, Any
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    evidence_text: str
    audit_result: Dict[str, Any]
    retry_count: int
    needs_web_search: bool