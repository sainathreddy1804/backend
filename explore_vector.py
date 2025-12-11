import os
import json
from weaviate import Client

# ───────────────────────────────
# Connect to Weaviate
# ───────────────────────────────
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
client = Client(WEAVIATE_URL)

# Check if Weaviate is reachable
try:
    if not client.is_ready():
        raise RuntimeError(f"Weaviate at {WEAVIATE_URL} is not ready")
    print(f"✅ Connected to Weaviate at {WEAVIATE_URL}")
except Exception as e:
    print(f"❌ Cannot connect to Weaviate: {e}")
    exit(1)

# ───────────────────────────────
# 1️⃣ List all classes in the schema
# ───────────────────────────────
try:
    schema = client.schema.get()
    classes = schema.get("classes", [])
    print("🗂 Classes in Weaviate:")
    for c in classes:
        print(f"- {c['class']}")
except Exception as e:
    print(f"❌ Error fetching schema: {e}")
    exit(1)

# ───────────────────────────────
# 2️⃣ Query objects from ArtEmbedding
# ───────────────────────────────
CLASS_NAME = "ArtEmbedding"  # Replace if your class name differs
FIELDS = ["filename", "style", "texture", "color", "emotion"]

try:
    response = client.query.get(CLASS_NAME, FIELDS).with_limit(5).do()  # get first 5 objects
    print("\n🖼 Sample objects from ArtEmbedding:")
    print(json.dumps(response, indent=2))
except Exception as e:
    print(f"❌ Error querying objects: {e}")
