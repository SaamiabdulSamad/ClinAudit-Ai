import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv

load_dotenv()

def initialize_medical_db():
    # Connect to your local Qdrant container
    client = QdrantClient("http://localhost:6333")
    
    collection_name = "medical_policies"
    
    # Check if collection already exists to avoid errors
    collections = client.get_collections().collections
    exists = any(c.name == collection_name for c in collections)
    
    if not exists:
        print(f"Creating collection: {collection_name}...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=1536,  # Standard for OpenAI embeddings
                distance=models.Distance.COSINE # Best for text similarity
            )
        )
        print("Collection created successfully!")
    else:
        print(f"ℹ️ Collection '{collection_name}' already exists. Skipping.")

if __name__ == "__main__":
    initialize_medical_db()