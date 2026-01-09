import fitz  # PyMuPDF
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid

# 1. Setup Client and Embeddings
client = QdrantClient("localhost", port=6333)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Ensure collection exists for "Clinical Protocols"
COLLECTION_NAME = "medicare_protocols"
if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )

def ingest_medical_policy(pdf_path: str, policy_category: str):
    """
    Parses a Medical Policy PDF and stores it with rich metadata for the Auditor.
    """
    doc = fitz.open(pdf_path)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )

    points = []
    
    for page_num, page in enumerate(doc):
        text = page.get_text()
        chunks = text_splitter.split_text(text)
        
        for chunk in chunks:
            # Generate embedding for the chunk
            vector = embeddings.embed_query(chunk)
            
            # Create a "Point" for Qdrant with Metadata
            # Metadata is what makes this "Enterprise Grade"
            point_id = str(uuid.uuid4())
            points.append(PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "content": chunk,
                    "metadata": {
                        "source": pdf_path,
                        "page": page_num + 1,
                        "category": policy_category,
                        "document_type": "Official Medicare Protocol"
                    }
                }
            ))
            
    # Batch upload to Qdrant
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"✅ Successfully ingested {len(points)} chunks from {pdf_path}")

# Example Usage:
# ingest_medical_policy("data/medicare_claims_manual.pdf", "Insurance Compliance")