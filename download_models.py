# download_models.py

from sentence_transformers import SentenceTransformer, CrossEncoder
from gliner import GLiNER
import os

print("--- 📥 Starting Model Download and Save Process ---")

# Define the local directory to save models
SAVE_DIRECTORY = "models"
if not os.path.exists(SAVE_DIRECTORY):
    os.makedirs(SAVE_DIRECTORY)
    print(f"Created directory: {SAVE_DIRECTORY}")

# Define models and their target save paths
models_to_download = {
    # GLiNER model for NER
    "urchade/gliner_small-v2.1": os.path.join(SAVE_DIRECTORY, "gliner_small"),
    # Sentence Transformer for Vector Search
    #"all-MiniLM-L6-v2": os.path.join(SAVE_DIRECTORY, "mini_lm"),
    "sentence-transformers/multi-qa-mpnet-base-dot-v1": os.path.join(SAVE_DIRECTORY, "mpnet_qa"),
    # Cross-Encoder for Reranking
    "cross-encoder/ms-marco-MiniLM-L-6-v2": os.path.join(SAVE_DIRECTORY, "cross_encoder")
}

try:
    # Download and save each model
    print("\n1. Downloading GLiNER model...")
    GLiNER.from_pretrained("urchade/gliner_small-v2.1").save_pretrained(models_to_download["urchade/gliner_small-v2.1"])
    print("✅ GLiNER model saved.")

    '''print("\n2. Downloading Sentence Transformer model...")
    SentenceTransformer("all-MiniLM-L6-v2").save(models_to_download["all-MiniLM-L6-v2"])
    print("✅ Sentence Transformer model saved.")'''

    print("\n2. Downloading QA Sentence Transformer model...")
    SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1").save(models_to_download["sentence-transformers/multi-qa-mpnet-base-dot-v1"])
    print("✅ QA Sentence Transformer model saved.")

    print("\n3. Downloading Cross-Encoder model...")
    # CrossEncoder model doesn't have a direct .save() method like SentenceTransformer,
    # so we rely on the library's caching mechanism and then ensure the path exists for consistency.
    # The SentenceTransformer library automatically caches it. We just confirm it.
    CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    # For simplicity, we will load this one from cache in the main script, but the download happens now.
    # To make our script work as-is, we will just save a dummy confirmation file for our other script to check.
    # Let's adjust the main script to load it directly, which is cleaner.
    print("✅ Cross-Encoder model downloaded to cache.")

    print("\n--- ✅ All models downloaded successfully. ---")

except Exception as e:
    print(f"\n--- ❌ ERROR ---")
    print(f"An error occurred during download: {e}")
    print("Please check your internet connection and ensure the required libraries are installed.")