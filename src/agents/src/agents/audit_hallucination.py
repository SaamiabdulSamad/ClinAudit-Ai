import os
from openai import OpenAI
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()
qdrant = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_KEY"))


def verify_faithfulness(query, answer, context_chunks):
    """
    The Senior Audit: Uses a 'Judge' model to detect unsupported claims.
    """
    context_text = "\n---\n".join(context_chunks)

    prompt = f"""
    You are a Medical Fact-Checker. 
    EVALUATION DATA:
    1. USER QUERY: {query}
    2. RETRIEVED PUBMED EVIDENCE: {context_text}
    3. AI-GENERATED ANSWER: {answer}

    TASK:
    Identify if the ANSWER contains "Hallucinations" (claims not supported by the EVIDENCE).
    
    OUTPUT FORMAT:
    - Status: [PASS/FAIL]
    - Hallucination Score: (0.0 to 1.0, where 1.0 is totally made up)
    - Unsupported Claims: [List any specific sentences that aren't in the evidence]
    """

    response = client.chat.completions.create(
        model="gpt-4o",  # Use a high-reasoning model for auditing
        messages=[{"role": "system", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content


test_query = "Does zinc reduce cold duration?"
mock_context = ["Study 102: Zinc acetate lozenges reduced cold duration by 3 days."]
mock_answer = "Zinc reduces cold duration by 3 days and also prevents hair loss."

print("🔍 Auditing for hallucinations...")
audit_report = verify_faithfulness(test_query, mock_answer, mock_context)
print(audit_report)
