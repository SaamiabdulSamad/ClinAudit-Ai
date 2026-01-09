from langchain_core.messages import AIMessage
from langchain_openai import OpenAIEmbeddings
from src.workflows.state import AgentState
from src.utils.vector_store import MedicalVectorStore
from src.schemas.custom_types import RAGSearchResult
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vs = MedicalVectorStore()

COLLECTION_NAME = "medicare_protocols"

def researcher_node(state: AgentState):
    """
    Researcher retrieves local policy evidence. 
    Implements a Circuit Breaker to prevent false denials when DB is down.
    """
    user_claim = state["messages"][0].content
    retry_count = state.get("retry_count", 0)
    
    print(f"\n🔍 RESEARCHER (Attempt {retry_count + 1})")

    # 🛑 THE CIRCUIT BREAKER: Check if Qdrant is alive before searching
    try:
        # We ping the vector store to ensure connection
        vs.client.get_collections() 
    except Exception as e:
        print(f"🔥 CRITICAL: Database unreachable ({e}). Switching to Web Escalation.")
        return {
            "messages": [AIMessage(content="⚠️ TECHNICAL ERROR: Local Database Offline.")],
            "evidence_text": "ERROR: DATABASE_OFFLINE",
            "needs_web_search": True # This triggers the Router to go to Tavily
        }

    # Increase recall on retry to catch specific CPT code tables
    top_k = 10 if retry_count > 0 else 5
    print(f"   → Searching collection '{COLLECTION_NAME}' (top_k={top_k})...")
    
    try:
        query_vector = embeddings.embed_query(user_claim)
        search_result: RAGSearchResult = vs.search(
            query_vector, 
            top_k=top_k, 
            collection_name=COLLECTION_NAME
        )
    except Exception as e:
        print(f"❌ Search Execution Failed: {e}")
        return {"evidence_text": "ERROR: SEARCH_FAILED", "needs_web_search": True}

    if search_result.contexts:
        # Priority sorting for 'noncovered' keywords
        combined_evidence = list(zip(search_result.contexts, search_result.sources))
        combined_evidence.sort(key=lambda x: ("noncovered" in x[0].lower() or "cpt" in x[0]), reverse=True)

        evidence_text = "### LOCAL POLICY EVIDENCE FOUND ###\n\n"
        for i, (context, source) in enumerate(combined_evidence):
            evidence_text += f"Source {i+1} [{source}]:\n{context}\n\n"
        
        print(f"   ✅ Found {len(search_result.contexts)} relevant chunks.")
        return {
            "messages": [AIMessage(content=evidence_text)],
            "retrieved_docs": [c for c, s in combined_evidence],
            "evidence_text": evidence_text,
            "retry_count": retry_count
        }
    
    print(f"   ❌ No policy found in '{COLLECTION_NAME}'")
    return {
        "messages": [AIMessage(content="⚠️ NO LOCAL POLICY FOUND.")],
        "retrieved_docs": [],
        "evidence_text": "",
        "needs_web_search": True # Escalate if local search is empty
    }