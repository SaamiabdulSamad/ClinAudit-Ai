import os
from qdrant_client import QdrantClient

# 1. SETUP - Use the exact same keys you put in AWS
URL = "https://your-cluster-id.us-east-1.aws.cloud.qdrant.io:6333" 
KEY = "your-long-api-key-string"

print(f"📡 Attempting to handshake with Qdrant at: {URL}")

try:
    # 2. INITßIALIZE CLIENT
    client = QdrantClient(url=URL, api_key=KEY)
    
    # 3. VERIFY CONNECTION
    collections = client.get_collections()
    print("✅ CONNECTION SUCCESSFUL!")
    print(f"📦 Existing Collections: {[c.name for c in collections.collections]}")

except Exception as e:
    print(f"❌ CONNECTION FAILED!")
    print(f"🔍 Error Detail: {e}")