import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from langchain_openai import OpenAIEmbeddings

# 1. Initialize
client = QdrantClient("localhost", port=6333)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
COLLECTION_NAME = "medicare_protocols"

def process_policy_directory(directory_path: str):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            path = os.path.join(directory_path, filename)
            loader = PyPDFLoader(path)
            
            # Extract Metadata from filename or header (Simplified here)
            # In a real system, use an LLM to extract these 4 fields from the first page
            metadata = {
                "policy_id": "L34555", 
                "jurisdiction": "Palmetto GBA",
                "document_type": "LCD",
                "source": filename
            }
            
            print(f"📦 Processing {filename} with Production Metadata...")
            pages = loader.load()
            chunks = text_splitter.split_documents(pages)
            
            points = []
            for i, chunk in enumerate(chunks):
                vector = embeddings.embed_query(chunk.page_content)
                points.append(PointStruct(
                    id=hash(f"{filename}_{i}"),
                    vector=vector,
                    payload={
                        "text": chunk.page_content,
                        "metadata": metadata, # THIS IS THE SCALE KEY
                        "page_number": chunk.metadata.get("page", 0)
                    }
                ))
            
            client.upsert(collection_name=COLLECTION_NAME, points=points)
    print("✅ Scale-ready Library Updated.")