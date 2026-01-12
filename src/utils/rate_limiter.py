import time
from functools import wraps

def medical_api_limiter(max_calls_per_min: int = 20):
    """
    Prevents API lockouts by throttling requests to LLM or 
    Tavily endpoints.
    """
    interval = 60.0 / max_calls_per_min
    last_call = [0.0]

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_call[0]
            if elapsed < interval:
                time.sleep(interval - elapsed)
            result = func(*args, **kwargs)
            last_call[0] = time.time()
            return result
        return wrapper
    return decorator