from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from langchain_openai import OpenAIEmbeddings

def init_qdrant_hybrid(collection_name: str):
    """
    Initializes a Qdrant store with Hybrid Search (Dense + Sparse/BM25) 
    to prevent 'Quiet Failures' in retrieval.
    """
# Use this for Production-Grade Docker setup
    client = QdrantClient(url="http://localhost:6333")    
    # Sparse embeddings for keyword/ID accuracy
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
    
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=OpenAIEmbeddings(),
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID # Combines both worlds
    )