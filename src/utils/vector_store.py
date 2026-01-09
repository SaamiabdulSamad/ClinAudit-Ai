import os
from typing import List, Optional
from dataclasses import dataclass
from qdrant_client import QdrantClient

@dataclass
class RAGSearchResult:
    contexts: List[str]
    sources: List[str]
    scores: List[float]

class MedicalVectorStore:
    def __init__(self):
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.client = QdrantClient(url=qdrant_url)
        # Default fallback, but we will override this in the search call
        self.default_collection = "medicare_protocols"
    
    def search(self, query_vector: List[float], top_k: int = 3, collection_name: Optional[str] = None) -> RAGSearchResult:
        """
        Search for clinical evidence. Now accepts a dynamic collection_name.
        """
        # Use the passed name, or fall back to the default
        target_collection = collection_name or self.default_collection
        
        try:
            # Check if the collection exists before searching to avoid 404s
            collections = self.client.get_collections().collections
            existing_names = [c.name for c in collections]
            
            if target_collection not in existing_names:
                print(f"⚠️ Warning: Collection '{target_collection}' not found. Available: {existing_names}")
                return RAGSearchResult(contexts=[], sources=[], scores=[])

            # Standard Qdrant Search
            results = self.client.search(
                collection_name=target_collection,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True
            )
            
            contexts = []
            sources = []
            scores = []
            
            for result in results:
                # Use .get() to safely access 'text' or 'page_content' depending on your ingestion script
                payload = result.payload
                text = payload.get('text') or payload.get('page_content') or ""
                source = payload.get('source') or payload.get('metadata', {}).get('source', 'Unknown')
                
                contexts.append(text)
                sources.append(source)
                scores.append(result.score)
            
            return RAGSearchResult(contexts=contexts, sources=sources, scores=scores)
            
        except Exception as e:
            print(f"⚠️ Vector search error in {target_collection}: {e}")
            return RAGSearchResult(contexts=[], sources=[], scores=[])