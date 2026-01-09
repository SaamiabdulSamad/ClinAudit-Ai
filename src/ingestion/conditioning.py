from langchain_core.documents import Document
import re

# Simple normalization mapping
MEDICAL_MAPPING = {
    r"\bhbp\b": "Hypertension",
    r"\bhigh blood pressure\b": "Hypertension",
    r"\bdm2\b": "Type 2 Diabetes Mellitus"
}

def normalize_medical_terms(text: str) -> str:
    """Replaces medical slang/shorthand with standardized terms."""
    for pattern, replacement in MEDICAL_MAPPING.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

def apply_conditioning(docs: list[Document]) -> list[Document]:
    """Applies cleaning and normalization before embedding."""
    for doc in docs:
        # 1. Clean whitespace and noise
        doc.page_content = " ".join(doc.page_content.split())
        # 2. Standardize terms
        doc.page_content = normalize_medical_terms(doc.page_content)
    return docs