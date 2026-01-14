import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

# This goes inside the .py file
URL = os.getenv("QDRANT_URL")
KEY = os.getenv("QDRANT_KEY")

print(f"📡 Attempting to handshake with Qdrant at: {URL}")

try:
    client = QdrantClient(url=URL, api_key=KEY)
    collections = client.get_collections()
    print("CONNECTION SUCCESSFUL!")
    print(f"Existing Collections: {[c.name for c in collections.collections]}")
except Exception as e:
    print(f"❌ CONNECTION FAILED: {e}")