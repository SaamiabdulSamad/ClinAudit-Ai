from dataclasses import dataclass
from typing import List

@dataclass
class RAGSearchResult:
    contexts: List[str]
    sources: List[str]
    scores: List[float]
