from sqlalchemy import create_engine, text

# 🔗 Update this with your actual Postgres connection string
DATABASE_URL = "postgresql+psycopg2://postgres:artpass@localhost:5432/art_search"

engine = create_engine(DATABASE_URL)

with engine.connect() as conn:
    conn.execute(text("ALTER TABLE artworks ADD COLUMN IF NOT EXISTS metadata_json JSONB;"))
    conn.execute(text("ALTER TABLE artworks ADD COLUMN IF NOT EXISTS search_count INTEGER DEFAULT 0;"))
    conn.commit()

print("✅ Columns added successfully!")