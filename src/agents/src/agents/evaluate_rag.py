import os, time, json, numpy as np
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
current_dir = os.path.dirname(os.path.abspath(__file__))
root_env = os.path.join(current_dir, "../../../../.env")
load_dotenv(root_env)

q_client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_KEY"))
o_client = OpenAI()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def is_semantic_match(query, retrieved_txt, ground_truth):
    """LLM-as-a-Judge: Semantic validation."""
    prompt = f"Query: {query}\nEvidence: {retrieved_txt}\nTruth: {ground_truth}\nDoes the Evidence confirm the Truth? Reply ONLY 'YES' or 'NO'."
    res = o_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": prompt}],
        max_tokens=2,
        temperature=0,
    )
    return "YES" in res.choices[0].message.content.upper()


print("📊 Starting Validated Metric Evaluation (n=50)...")
res = q_client.scroll(collection_name="pubmed_docs", limit=50, with_payload=True)
points = res[0]

total_queries = len(points)
tp_count = 0  # Total relevant chunks found
queries_successful = 0  # Queries where at least 1 relevant chunk was found
total_retrieved = 0
search_latencies = []

for point in tqdm(points):
    ground_truth = point.payload.get("page_content", point.payload.get("text", ""))
    if not ground_truth:
        continue

    query_text = f"Provide a technical summary of: {ground_truth[:100]}"
    vector = embeddings.embed_query(query_text)

    start = time.time()
    response = q_client.query_points(
        collection_name="pubmed_docs", query=vector, limit=5, with_payload=True
    )
    search_latencies.append(time.time() - start)

    hits = response.points
    found_truth_for_this_query = False

    for hit in hits:
        total_retrieved += 1
        retrieved_txt = hit.payload.get("page_content", hit.payload.get("text", ""))

        if is_semantic_match(query_text, retrieved_txt, ground_truth):
            tp_count += 1
            found_truth_for_this_query = True

    if found_truth_for_this_query:
        queries_successful += 1

precision = tp_count / total_retrieved if total_retrieved > 0 else 0
recall = queries_successful / total_queries if total_queries > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
p95_lat = np.percentile(search_latencies, 95) * 1000

print(
    f"""
--- VALIDATED EVALUATION REPORT ---
Precision@5: {precision:.2f}  (Signal-to-Noise Ratio)
Recall@5:    {recall:.2f}     (Search Success Rate - MUST BE 0-1)
F1 Score:    {f1:.2f}         (Retrieval Balance)
P95 Latency: {p95_lat:.2f}ms    (True Qdrant Speed)
----------------------------------
"""
)
