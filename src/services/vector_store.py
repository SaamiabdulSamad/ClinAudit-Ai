from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient

def get_vector_store():
    client = QdrantClient(path="data/qdrant_db")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    return QdrantVectorStore(
        client=client,
        collection_name="medical_knowledge",
        embedding=embeddings
    )