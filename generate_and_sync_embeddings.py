import os
import json
import uuid
import time
import logging
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy.orm import Session
import numpy as np

from database import SessionLocal
from models import Artwork, Embedding
from ml_models import generate_all_embeddings
from main import auto_tags_from_embeddings
from vector_db import insert_embedding_to_weaviate, get_weaviate_client

# ─────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("embedding_sync_parallel.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("embedding_sync_parallel")


# ─────────────────────────────────────────────
# Helper: safe conversion
# ─────────────────────────────────────────────
def to_list(v):
    if isinstance(v, list):
        return v
    elif hasattr(v, "tolist"):
        return v.tolist()
    else:
        return np.asarray(v, dtype=np.float32).tolist()


# ─────────────────────────────────────────────
# Worker function (robust + atomic)
# ─────────────────────────────────────────────
def process_artwork(art):
    """Generate embeddings + metadata safely and sync both Postgres and Weaviate."""
    session = SessionLocal()
    try:
        if not art.filepath or not os.path.exists(art.filepath):
            logger.warning(f"⚠️ Skipped (missing path): {art.filename}")
            return f"⚠️ Skipped: {art.filename}"

        # ✅ Generate embeddings (with retries)
        embs = None
        for attempt in range(2):
            try:
                embs = generate_all_embeddings(art.filepath)
                break
            except Exception as e:
                logger.warning(f"Retry {attempt+1}/2 for {art.filename}: {e}")
                time.sleep(2 ** attempt)
        if not embs:
            raise RuntimeError(f"❌ Embedding generation failed for {art.filename}")

        # ✅ Generate AI tags
        tags = auto_tags_from_embeddings(embs)

        # ✅ Update Artwork metadata
        art.style = tags.get("style", "")
        art.color = tags.get("color", "")
        art.texture = tags.get("texture", "")
        art.emotion = tags.get("emotion", "")
        art.metadata_json = json.dumps(tags or {}, ensure_ascii=False)

        # ✅ Upsert embedding record
        emb_row = session.query(Embedding).filter(Embedding.artwork_id == art.id).first()
        if not emb_row:
            emb_row = Embedding(artwork_id=art.id)
            session.add(emb_row)

        emb_row.style_vector = json.dumps(to_list(embs.get("style", [])))
        emb_row.color_vector = json.dumps(to_list(embs.get("palette", embs.get("color", []))))
        emb_row.texture_vector = json.dumps(to_list(embs.get("texture", [])))
        emb_row.emotion_vector = json.dumps(to_list(embs.get("emotion", [])))
        emb_row.vector = json.dumps(to_list(embs.get("style", [])))  # fallback
        emb_row.embedding_model = "CLIP-ViT-B-32 + DINOv2-Small"
        emb_row.embedding_dim = 512

        # ✅ Commit DB changes before Weaviate push
        session.commit()

        # ✅ Deterministic UUID
        obj_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(art.id)))

        # ✅ Push to Weaviate (up to 3 retries)
        for attempt in range(3):
            try:
                insert_embedding_to_weaviate(art, embs, obj_uuid)
                break
            except Exception as e:
                logger.warning(f"⏳ Retry {attempt+1}/3 Weaviate push for {art.filename}: {e}")
                time.sleep(2 ** attempt)

        return f"✅ Synced: {art.filename}"

    except Exception as e:
        session.rollback()
        logger.exception(f"❌ Error processing {art.filename}: {e}")
        return f"❌ Failed: {art.filename}"

    finally:
        session.close()


# ─────────────────────────────────────────────
# Batch-wise Parallel Runner
# ─────────────────────────────────────────────
def regenerate_and_sync_parallel(mode="full", workers=8, batch_size=1000, start_offset=0):
    """Main parallel runner with full or missing-only modes."""
    db = SessionLocal()

    # 🧠 Mode selection
    if mode == "missing-only":
        artworks_query = (
            db.query(Artwork)
            .filter(Artwork.is_permanent == True)
            .filter(~Artwork.id.in_(db.query(Embedding.artwork_id)))
        )
        logger.info("🟡 Mode: Missing-only → will process only artworks without embeddings.")
    else:
        artworks_query = db.query(Artwork).filter(Artwork.is_permanent == True)
        logger.info("🟢 Mode: Full → will regenerate all embeddings and metadata.")

    total = artworks_query.count()
    logger.info(f"🎨 Total artworks to process: {total}")

    # Ensure Weaviate is ready before starting
    get_weaviate_client()

    for offset in range(start_offset, total, batch_size):
        batch = (
            artworks_query.order_by(Artwork.id)
            .offset(offset)
            .limit(batch_size)
            .all()
        )

        valid = [a for a in batch if os.path.exists(a.filepath)]
        if not valid:
            logger.info(f"⏭️ Empty or invalid batch (offset={offset})")
            continue

        logger.info(f"🚀 Batch {offset // batch_size + 1}: {len(valid)} artworks")

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_artwork, art): art.filename for art in valid}
            for f in tqdm(as_completed(futures), total=len(valid),
                          desc=f"Batch {offset // batch_size + 1}", unit="artwork"):
                try:
                    msg = f.result()
                    if msg:
                        logger.info(msg)
                except Exception as e:
                    logger.error(f"⚠️ Worker error: {e}")

        logger.info(f"✅ Batch {offset // batch_size + 1} complete.\n")

    db.close()
    logger.info("🎉 Full re-sync completed successfully.")


# ─────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate and sync embeddings for artworks.")
    parser.add_argument("--mode", choices=["full", "missing-only"], default="full",
                        help="Choose whether to regenerate all or only missing embeddings.")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers.")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size.")
    args = parser.parse_args()

    try:
        regenerate_and_sync_parallel(
            mode=args.mode,
            workers=args.workers,
            batch_size=args.batch_size
        )
    except KeyboardInterrupt:
        logger.warning("🛑 Interrupted by user. Exiting gracefully.")
