from database import Base, engine
from models import *

print("🧱 Creating tables in PostgreSQL database...")
Base.metadata.create_all(bind=engine)
print("✅ Tables created successfully!")