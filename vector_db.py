import os
import time
import json
import uuid
import numpy as np
import logging
from weaviate import Client
from weaviate.exceptions import WeaviateBaseError

# ───────────────────────────────
# Logging setup
# ───────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("weaviate")

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://weaviate:8080").rstrip("/")

# ───────────────────────────────
# Connect to Weaviate
# ───────────────────────────────
def get_weaviate_client(retries: int = 15, delay: int = 5) -> Client:
    """Try connecting with retries."""
    for attempt in range(1, retries + 1):
        try:
            client = Client(WEAVIATE_URL)
            client.schema.get()
            logger.info(f"✅ Connected to Weaviate at {WEAVIATE_URL}")
            return client
        except Exception as e:
            logger.warning(f"⏳ Waiting for Weaviate (attempt {attempt}/{retries}): {e}")
            time.sleep(delay)
    raise RuntimeError("❌ Could not connect to Weaviate.")

# ───────────────────────────────
# Schema setup (single vector)
# ───────────────────────────────
def init_weaviate_schema():
    """Create ArtEmbedding class if missing."""
    class_name = "ArtEmbedding"
    if not client.schema.exists(class_name):
        schema = {
            "class": class_name,
            "description": "Artwork metadata + concatenated 4-feature embedding vector",
            "vectorizer": "none",
            "properties": [
                {"name": "filename", "dataType": ["text"]},
                {"name": "filepath", "dataType": ["text"]},
                {"name": "title", "dataType": ["text"]},
                {"name": "artist", "dataType": ["text"]},
                {"name": "style", "dataType": ["text"]},
                {"name": "color", "dataType": ["text"]},
                {"name": "texture", "dataType": ["text"]},
                {"name": "emotion", "dataType": ["text"]},
                {"name": "metadata_json", "dataType": ["text"]},
                {"name": "is_permanent", "dataType": ["boolean"]},
            ],
        }
        client.schema.create_class(schema)
        logger.info("✅ Created Weaviate schema 'ArtEmbedding'")
    else:
        logger.info("ℹ️ Weaviate class 'ArtEmbedding' already exists")

# ───────────────────────────────
# Insert / update function
# ───────────────────────────────
def insert_embedding_to_weaviate(artwork, embs: dict, object_id: str):
    """
    Insert or update one artwork with its 4 embeddings (merged into one 2048-D vector)
    and full metadata.
    """
    try:
        # Handle palette → color
        if "palette" in embs and "color" not in embs:
            embs["color"] = embs["palette"]

        # Combine 4 vectors into one
        full_vector = np.concatenate([
            np.array(embs["style"], dtype=np.float32),
            np.array(embs["color"], dtype=np.float32),
            np.array(embs["texture"], dtype=np.float32),
            np.array(embs["emotion"], dtype=np.float32),
        ]).astype(np.float32).tolist()

        # Metadata payload
        data = {
            "filename": artwork.filename,
            "filepath": artwork.filepath,
            "title": artwork.title or "",
            "artist": artwork.artist or "",
            "style": artwork.style or "Unknown",
            "color": artwork.color or "Unknown",
            "texture": artwork.texture or "Unknown",
            "emotion": artwork.emotion or "Unknown",
            "metadata_json": json.dumps({
                "style": artwork.style,
                "color": artwork.color,
                "texture": artwork.texture,
                "emotion": artwork.emotion,
            }),
            "is_permanent": True,
        }

        # Replace existing object or create new
        try:
            client.data_object.replace(
                uuid=object_id,
                class_name="ArtEmbedding",
                data_object=data,
                vector=full_vector,
            )
            logger.info(f"♻️ Replaced existing: {artwork.filename}")
        except WeaviateBaseError:
            client.data_object.create(
                class_name="ArtEmbedding",
                uuid=object_id,
                data_object=data,
                vector=full_vector,
            )
            logger.info(f"✨ Created new: {artwork.filename}")

    except Exception as e:
        logger.error(f"❌ Failed for {artwork.filename}: {e}")


# Initialize client and schema
client = get_weaviate_client()
init_weaviate_schema()
