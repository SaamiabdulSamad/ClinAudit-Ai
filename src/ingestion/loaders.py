# The line `from langchain_community.document_loaders import PyPDFLoader` is importing the
# `PyPDFLoader` class from the `document_loaders` module within the `langchain_community` package.
# This class is used for loading PDF documents and extracting their content for further processing in
# the script.
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from datetime import datetime
import os

def load_medical_pdf(file_path: str) -> list[Document]:
    """
    Loads a PDF and injects critical medical metadata for 
    production-grade filtering in Qdrant.
    """
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    # Enrichment: Add metadata for 'Freshness' and 'Source Authority'
    for doc in docs:
        doc.metadata.update({
            "source_path": file_path,
            "ingestion_date": datetime.now().isoformat(),
            "document_type": "Clinical_Note" if "note" in file_path.lower() else "Medical_Research",
            "access_level": "Restricted"
        })
    return docs
