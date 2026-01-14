import os
import time
from datasets import load_dataset
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
from tqdm import tqdm  # For that professional progress bar

load_dotenv(os.path.join(os.path.dirname(__file__), "../../../../.env"))

COLLECTION_NAME = "pubmed_docs"
BATCH_SIZE = 100  # Senior move: Batching prevents API timeouts

client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_KEY"))
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
    quantization_config=models.ScalarQuantization(
        scalar=models.ScalarQuantizationConfig(
            type=models.ScalarType.INT8, quantile=0.99, always_ram=True
        )
    ),
)

print("Starting High-Volume Ingestion (Target: ~25,000+ chunks)...")
dataset = load_dataset(
    "ccdv/pubmed-summarization", "section", split="train", streaming=True
)
papers = list(dataset.take(500))  # 500 papers ≈ 25k-30k chunks

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
all_docs = []

for i, paper in enumerate(papers):
    chunks = text_splitter.split_text(paper.get("article", ""))
    for chunk in chunks:
        all_docs.append({"text": chunk, "metadata": {"source": f"pubmed_{i}"}})

print(f"Total Chunks: {len(all_docs)}. Ingesting in batches...")
for i in tqdm(range(0, len(all_docs), BATCH_SIZE)):
    batch = all_docs[i : i + BATCH_SIZE]
    QdrantVectorStore.from_texts(
        texts=[d["text"] for d in batch],
        embedding=embeddings,
        metadatas=[d["metadata"] for d in batch],
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_KEY"),
        collection_name=COLLECTION_NAME,
    )

print(f"SUCCESS: {len(all_docs)} shards live with Optimized Latency.")
