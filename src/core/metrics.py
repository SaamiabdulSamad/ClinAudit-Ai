def calculate_faithfulness_score(claims, context_chunks):
    """
    Calculates how many claims are backed by the retrieved medical data.
    """
    if not claims:
        return 0.0
    
    supported_count = sum(1 for claim in claims if claim.is_supported)
    return supported_count / len(claims)

def calculate_hallucination_rate(total_audits, failed_audits):
    """
    Standard MLOps metric for tracking agent reliability.
    """
    if total_audits == 0:
        return 0.0
    return (failed_audits / total_audits) * 100
