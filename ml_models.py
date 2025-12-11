import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import open_clip


# ─────────────────────────────────────────────
# Device configuration
# ─────────────────────────────────────────────
DEVICE = "cpu"  # change to "cuda" if GPU available

# ─────────────────────────────────────────────
# Load models once
# ─────────────────────────────────────────────

# CLIP model → used for style, color, and emotion
CLIP_MODEL, _, CLIP_PREPROCESS = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
CLIP_MODEL = CLIP_MODEL.to(DEVICE)

# DINOv2 model → used for texture
DINO_PROCESSOR = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
DINO_MODEL = AutoModel.from_pretrained("facebook/dinov2-small").to(DEVICE)


# ─────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────
def _normalize(vec):
    """Normalize embedding vector to unit length."""
    v = np.asarray(vec, dtype=np.float32)
    n = np.linalg.norm(v)
    return (v / n).tolist() if n > 0 else v.tolist()


def _fallback_rgb_embedding(image_path):
    """Fallback color embedding using mean/std RGB values."""
    try:
        img = Image.open(image_path).convert("RGB").resize((100, 100))
        arr = np.array(img, dtype=np.float32)
        mean_rgb = arr.mean(axis=(0, 1)) / 255.0
        std_rgb = arr.std(axis=(0, 1)) / 255.0
        base = np.concatenate([mean_rgb, std_rgb])
        base = np.pad(base, (0, 512 - len(base)))
        return _normalize(base)
    except Exception as e:
        print(f"⚠️ Fallback RGB embedding failed for {image_path}: {e}")
        return np.zeros(512, dtype=np.float32).tolist()


# ─────────────────────────────────────────────
# 1️⃣ STYLE → CLIP embedding
# ─────────────────────────────────────────────
def generate_style_embedding(image_path):
    """Generate semantic style embedding using CLIP."""
    try:
        image = CLIP_PREPROCESS(Image.open(image_path).convert("RGB")).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = CLIP_MODEL.encode_image(image)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return _normalize(emb.cpu().numpy().flatten())
    except Exception as e:
        print(f"⚠️ Style embedding failed for {image_path}: {e}")
        return np.zeros(512, dtype=np.float32).tolist()


# ─────────────────────────────────────────────
# 2️⃣ TEXTURE → DINOv2 embedding (padded to 512D)
# ─────────────────────────────────────────────
def generate_texture_embedding(image_path):
    """Generate texture embedding using DINOv2 (surface patterns), padded to 512D."""
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = DINO_PROCESSOR(images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = DINO_MODEL(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        vec = emb.cpu().numpy().flatten()

        # Pad to match 512D schema
        vec = np.pad(vec, (0, 512 - len(vec)))  
        return _normalize(vec)
    except Exception as e:
        print(f"⚠️ Texture embedding failed for {image_path}: {e}")
        return np.zeros(512, dtype=np.float32).tolist()


# ─────────────────────────────────────────────
# 3️⃣ COLOR / PALETTE → CLIP-based + fallback
# ─────────────────────────────────────────────
def generate_palette_embedding(image_path):
    """
    Generate AI-based color palette embedding using CLIP.
    Uses conceptual color prompts, falls back to RGB stats on failure.
    """
    try:
        color_prompts = [
            "warm colors", "cool colors", "neutral tones",
            "pastel palette", "vibrant colors", "monochrome style",
            "earth tones", "dark tones", "bright tones", "muted palette"
        ]

        image = CLIP_PREPROCESS(Image.open(image_path).convert("RGB")).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            img_emb = CLIP_MODEL.encode_image(image)
            txt_embs = CLIP_MODEL.encode_text(open_clip.tokenize(color_prompts).to(DEVICE))

        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        txt_embs = txt_embs / txt_embs.norm(dim=-1, keepdim=True)

        sims = (img_emb @ txt_embs.T).cpu().numpy().flatten()
        vec = np.pad(sims, (0, 512 - len(sims)))
        return _normalize(vec)

    except Exception as e:
        print(f"⚠️ Color embedding failed for {image_path}: {e}")
        return _fallback_rgb_embedding(image_path)


# ─────────────────────────────────────────────
# 4️⃣ EMOTION → CLIP text-guided similarity
# ─────────────────────────────────────────────
def generate_emotion_embedding(image_path):
    """
    Generate emotion embedding using CLIP text prompts.
    Embedding represents similarity to emotion concepts.
    """
    try:
        emotions = [
            "calm", "joyful", "energetic", "melancholic",
            "mysterious", "romantic", "serene", "sad", "hopeful", "tense"
        ]

        image = CLIP_PREPROCESS(Image.open(image_path).convert("RGB")).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            img_emb = CLIP_MODEL.encode_image(image)
            txt_embs = CLIP_MODEL.encode_text(open_clip.tokenize(emotions).to(DEVICE))

        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        txt_embs = txt_embs / txt_embs.norm(dim=-1, keepdim=True)

        sims = (img_emb @ txt_embs.T).cpu().numpy().flatten()
        vec = np.pad(sims, (0, 512 - len(sims)))
        return _normalize(vec)

    except Exception as e:
        print(f"⚠️ Emotion embedding failed for {image_path}: {e}")
        return np.zeros(512, dtype=np.float32).tolist()


# ─────────────────────────────────────────────
# Optional unified function
# ─────────────────────────────────────────────
def generate_all_embeddings(image_path):
    """Generate all embeddings for one image."""
    return {
        "style": generate_style_embedding(image_path),
        "texture": generate_texture_embedding(image_path),
        "color": generate_palette_embedding(image_path),
        "emotion": generate_emotion_embedding(image_path)
    }


# ─────────────────────────────────────────────
# Validation helper (optional)
# ─────────────────────────────────────────────
def validate_embedding_shapes(image_path):
    """Quick check that all embeddings are 512D before inserting to DB."""
    embs = generate_all_embeddings(image_path)
    for name, vec in embs.items():
        print(f"{name:8s} → {len(vec)} dimensions")
    return all(len(v) == 512 for v in embs.values())


# ─────────────────────────────────────────────
# Demo run
# ─────────────────────────────────────────────
if __name__ == "__main__":
    test_path = "data/images/example_art.jpg"  # replace with your test image
    all_embs = generate_all_embeddings(test_path)

    print("\n✅ Embeddings generated successfully:")
    for k, v in all_embs.items():
        print(f"• {k.capitalize():8s}: len={len(v)}  sample={v[:5]}")

    print("\nValidation:", validate_embedding_shapes(test_path))
