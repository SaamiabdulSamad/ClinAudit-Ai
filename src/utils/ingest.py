import uuid
import os
from openai import OpenAI
from src.utils.data_loader import MedicalDataLoader
from src.utils.vector_store import MedicalVectorStore
from dotenv import load_dotenv

# Load environment variables (API Keys)
load_dotenv()

def run_ingestion():
    # 1. Initialize our Tech With Tim style tools
    client = OpenAI()
    loader = MedicalDataLoader(chunk_size=1000, chunk_overlap=200)
    vs = MedicalVectorStore()

    # Path to your medical PDF
    pdf_path = "data/medicare_policy.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ Error: Could not find {pdf_path}. Ensure it's in the 'data' folder.")
        return

    # 2. Extract and Chunk
    print("📖 Reading and chunking medical policy...")
    chunks = loader.load_and_chunk_pdf(pdf_path)

    # 3. Generate High-Accuracy Embeddings
    print(f"Embedding {len(chunks)} chunks with 'text-embedding-3-large'...")
    embeddings_response = client.embeddings.create(
        input=chunks,
        model="text-embedding-3-large"
    )

    # 4. Prepare for Qdrant
    ids = [str(uuid.uuid4()) for _ in chunks]
    vectors = [e.embedding for e in embeddings_response.data]
    
    # Payload matches the schema for the Auditor to cite evidence
    payloads = [
        {
            "text": chunk_text, 
            "source": os.path.basename(pdf_path),
            "type": "medical_policy"
        } 
        for chunk_text in chunks
    ]

    # 5. Push to Local Qdrant
    print("Uploading evidence to Qdrant...")
    vs.upsert(ids=ids, vectors=vectors, payloads=payloads)
    
    print(f"Success! Your Auditor now has {len(chunks)} verified medical rules.")

if __name__ == "__main__":
    run_ingestion()