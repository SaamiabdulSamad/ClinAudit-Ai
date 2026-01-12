from prometheus_client import Counter, Histogram, Gauge, Summary

# 1. Track the 'Faithfulness' score from your Auditor
AUDIT_FAITHFULNESS_SCORE = Gauge(
    "factguard_faithfulness_score", 
    "Faithfulness score (0.0 - 1.0) of the last audit",
    ["claim_type"]
)

# 2. Count Hallucinations (Scores below 0.7)
HALLUCINATION_COUNT = Counter(
    "factguard_hallucinations_total",
    "Total number of suspected hallucinations detected"
)

# 3. Track Latency of the entire Multi-Agent Loop
AGENT_LOOP_LATENCY = Histogram(
    "factguard_agent_latency_seconds",
    "Time spent in the researcher-auditor loop",
    buckets=(1.0, 2.0, 5.0, 10.0, 30.0)
)

# 4. Track Token Usage (Cost Control)
TOKEN_USAGE_COUNTER = Counter(
    "factguard_tokens_total",
    "Total tokens consumed by agents",
    ["model_name"]
)