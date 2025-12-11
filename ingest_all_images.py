import os
from database import SessionLocal
from models import Artwork

def ingest_all_images(base_dirs=["images", "data"]):
    """
    Recursively scan given directories for image files
    and insert missing ones into the artworks table.
    """
    db = SessionLocal()
    existing = {a.filename for a in db.query(Artwork.filename).all()}
    added = 0
    scanned = 0

    for base_dir in base_dirs:
        for root, _, files in os.walk(base_dir):
            for filename in files:
                scanned += 1
                if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    continue
                if filename in existing:
                    continue

                filepath = os.path.join(root, filename)
                if not os.path.exists(filepath):
                    continue

                db.add(Artwork(
                    filename=filename,
                    filepath=filepath,
                    is_permanent=True
                ))
                added += 1

    db.commit()
    db.close()
    print(f"✅ Scanned {scanned} files; inserted {added} new artworks.")

if __name__ == "__main__":
    ingest_all_images()
