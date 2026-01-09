import json
import asyncio
import pandas as pd
from datetime import datetime
from src.agents.graph import app as agent_graph
from langchain_core.messages import HumanMessage

# 1. THE GOLD STANDARD TEST SET
# These are designed to test if your Auditor catches hallucinations vs. facts.
TEST_CASES = [
    {
        "id": "TC_001",
        "claim": "Does Medicare Part B cover outpatient physical therapy?",
        "expected_verdict": "PASS",
        "category": "Coverage Inquiry"
    },
    {
        "id": "TC_002",
        "claim": "Is the experimental drug 'Zylophin' covered for stage 4 lung cancer under standard CMS policy?",
        "expected_verdict": "FAIL", # Should fail because it's experimental/not in policy
        "category": "Experimental Drug"
    },
    {
        "id": "TC_003",
        "claim": "Can a patient receive a heart transplant if they are currently an active smoker according to UNOS guidelines?",
        "expected_verdict": "FAIL", # Should flag as needs review/contradicted
        "category": "Clinical Eligibility"
    }
]

async def run_evaluation():
    results = []
    print(f"🚀 Starting FactGuard Evaluation at {datetime.now()}")
    print("-" * 50)

    for case in TEST_CASES:
        print(f"Testing {case['id']}: {case['category']}...")
        
        start_time = asyncio.get_event_loop().time()
        
        # Invoke the multi-agent graph
        response = await agent_graph.ainvoke({
            "messages": [HumanMessage(content=case["claim"])],
            "patient_claim": case["claim"]
        })
        
        end_time = asyncio.get_event_loop().time()
        
        # Extract metrics
        actual_verdict = response.get("verdict", "UNKNOWN")
        score = response.get("faithfulness_score", 0.0)
        
        results.append({
            "ID": case["id"],
            "Claim": case["claim"],
            "Expected": case["expected_verdict"],
            "Actual": actual_verdict,
            "Success": "✅" if actual_verdict == case["expected_verdict"] else "❌",
            "Faithfulness": score,
            "Latency": round(end_time - start_time, 2)
        })

    # 2. GENERATE REPORT
    df = pd.DataFrame(results)
    avg_faithfulness = df["Faithfulness"].mean()
    accuracy = (df["Expected"] == df["Actual"]).mean() * 100

    print("\n" + "="*50)
    print("FINAL EVALUATION REPORT")
    print("="*50)
    print(df.to_markdown(index=False))
    print(f"\n📈 Overall Accuracy: {accuracy:.1f}%")
    print(f"⭐ Average Faithfulness: {avg_faithfulness:.2f}")
    
    # Save to CSV for your portfolio
    df.to_csv("eval_report_latest.csv", index=False)
    print("\n✅ Report saved to eval_report_latest.csv")

if __name__ == "__main__":
    asyncio.run(run_evaluation())
    TEST_CASES = [
    {
        "id": "PROD_TEST_001",
        "claim": "Is CPT code 0058T (Cryopreservation of ovary tissue) covered?",
        "expected_verdict": "FAIL" 
    },
    {
        "id": "PROD_TEST_002",
        "claim": "Patient requires Ocular blood flow measurement (CPT 0198T). Will Medicare pay?",
        "expected_verdict": "FAIL"
    }
]