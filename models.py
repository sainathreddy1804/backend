from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Float, Boolean
from sqlalchemy.sql import func
from sqlalchemy.types import JSON
from database import Base
from datetime import datetime
from sqlalchemy.orm import relationship

# ───────────────────────────────
# User table
# ───────────────────────────────
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, default="user")

# ───────────────────────────────
# Artwork table
# ───────────────────────────────
class Artwork(Base):
    __tablename__ = "artworks"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    filepath = Column(String, nullable=False)

    title = Column(String, nullable=True)
    artist = Column(String, nullable=True)
    style = Column(String, nullable=True)
    color = Column(String, nullable=True)
    texture = Column(String, nullable=True)
    emotion = Column(String, nullable=True)

    metadata_json = Column(JSON, nullable=True)
    is_permanent = Column(Boolean, default=True)
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())

    # 🔹 Optional tracking fields
    search_count = Column(Integer, default=0)  # how often this artwork is returned in search
    last_synced_at = Column(DateTime, default=datetime.utcnow)  # when embeddings were last updated

    embedding = relationship(
        "Embedding",
        back_populates="artwork",
        uselist=False,
        cascade="all, delete-orphan"
    )


# ───────────────────────────────
# Embedding table (AI-powered)
# ───────────────────────────────
class Embedding(Base):
    __tablename__ = "embeddings"

    id = Column(Integer, primary_key=True, index=True)
    artwork_id = Column(Integer, ForeignKey("artworks.id", ondelete="CASCADE"), unique=True, nullable=False)

    # Embedding vectors
    vector = Column(JSON, nullable=False)  # Combined or general-purpose vector
    style_vector = Column(JSON, nullable=True)
    color_vector = Column(JSON, nullable=True)
    texture_vector = Column(JSON, nullable=True)
    emotion_vector = Column(JSON, nullable=True)

    # Model info
    embedding_model = Column(String, nullable=True)
    embedding_dim = Column(Integer, default=512)
    model_version = Column(String, default="CLIP-ViT-B-32 + DINOv2-Small")
    embedding_source = Column(String, default="AI")

    # Maintenance fields
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    artwork = relationship("Artwork", back_populates="embedding")



# ───────────────────────────────
# Metadata validation table
# ───────────────────────────────
class MetadataValidation(Base):
    __tablename__ = "metadata_validation"

    id = Column(Integer, primary_key=True)
    artwork_id = Column(Integer, ForeignKey("artworks.id"))
    status = Column(String)  # PASSED / FAILED
    issues = Column(Text, nullable=True)
    timestamp = Column(DateTime)


# ───────────────────────────────
# Data quality audit table
# ───────────────────────────────
class DataQualityAudit(Base):
    __tablename__ = "data_quality_audit"

    id = Column(Integer, primary_key=True, index=True)
    total_embeddings = Column(Integer)
    valid_embeddings = Column(Integer)
    invalid_embeddings = Column(Integer)
    avg_norm = Column(Float)
    norm_std = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="OK")
