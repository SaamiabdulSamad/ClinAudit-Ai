import os
from qdrant_client import QdrantClient
from qdrant_client.http import models

client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_KEY"))

client.recreate_collection(
    collection_name="pubmed_docs",
    vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
)
print(" Collection 'pubmed_docs' created!")
