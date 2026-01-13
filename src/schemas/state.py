from typing import Annotated, TypedDict, List
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    audit_score: float
    needs_revision: bool