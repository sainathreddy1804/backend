import os
import json
import numpy as np
from weaviate import Client

# ───────────────────────────────
# Connect to Weaviate
# ───────────────────────────────
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
client = Client(WEAVIATE_URL)

if not client.is_ready():
    raise RuntimeError(f"Weaviate at {WEAVIATE_URL} is not ready")
print(f"✅ Connected to Weaviate at {WEAVIATE_URL}")

# ───────────────────────────────
# Query all objects from ArtEmbedding
# ───────────────────────────────
CLASS_NAME = "ArtEmbedding"
FIELDS = ["filename", "style", "texture", "color", "emotion", "metadata_json"]

LIMIT = 50  # adjust as needed

response = client.query.get(CLASS_NAME, FIELDS) \
    .with_additional(["vector"]) \
    .with_limit(LIMIT) \
    .do()

objects = response.get("data", {}).get("Get", {}).get(CLASS_NAME, [])

if not objects:
    print("⚠️ No objects found in ArtEmbedding.")
else:
    print(f"🖼 Found {len(objects)} objects in {CLASS_NAME}:\n")

    for obj in objects:
        props = obj.get("properties", {})
        vector = obj.get("_additional", {}).get("vector", [])

        print(f"Filename: {props.get('filename')}")
        print(f"Vector length (combined): {len(vector)}")

        # Try to parse metadata_json if exists
        metadata_str = props.get("metadata_json")
        if metadata_str:
            try:
                metadata = json.loads(metadata_str)
                print("🔹 Detailed embeddings / metadata:")
                for key in ["style", "texture", "color", "emotion"]:
                    val = metadata.get(key)
                    if isinstance(val, list):
                        print(f"  {key} vector (length {len(val)}): {val[:5]} ...")  # print first 5 values
                    else:
                        print(f"  {key}: {val}")
            except Exception as e:
                print(f"⚠️ Could not parse metadata_json: {e}")
        else:
            print("⚠️ No metadata_json found")

        print("-" * 50)

    # Optional: save combined vectors and metadata for later
    embeddings_list = [obj.get("_additional", {}).get("vector", []) for obj in objects]
    metadata_list = [obj.get("properties", {}) for obj in objects]

    np.save("art_embeddings.npy", np.array(embeddings_list))
    with open("art_metadata.json", "w") as f:
        json.dump(metadata_list, f, indent=2)

    print("\n✅ Saved embeddings to 'art_embeddings.npy'")
    print("✅ Saved metadata to 'art_metadata.json'")
