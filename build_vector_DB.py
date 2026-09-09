import json
import os
import faiss
import numpy as np


INPUT_FILE = "chunks_with_embeddings.json"
OUTPUT_DIR = "vector_db"

INDEX_FILE = os.path.join(OUTPUT_DIR, "index.faiss")
METADATA_FILE = os.path.join(OUTPUT_DIR, "metadata.json")


# --------------------------------------------------
# 1. Load embedded chunks
# --------------------------------------------------

print("Loading chunks...")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Total chunks loaded: {len(chunks)}")


# --------------------------------------------------
# 2. Extract embeddings
# --------------------------------------------------

embeddings = np.array(
    [chunk["embedding"] for chunk in chunks],
    dtype="float32"
)

print(f"Embedding shape: {embeddings.shape}")


# --------------------------------------------------
# 3. Create FAISS index
# --------------------------------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

print("Adding embeddings to FAISS...")

index.add(embeddings)

print(f"Vectors stored in FAISS: {index.ntotal}")


# --------------------------------------------------
# 4. Create output directory
# --------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# 5. Save FAISS index
# --------------------------------------------------

faiss.write_index(index, INDEX_FILE)

print(f"FAISS index saved to: {INDEX_FILE}")


# --------------------------------------------------
# 6. Save metadata
# --------------------------------------------------

metadata = []

for chunk in chunks:
    metadata.append({
        "text": chunk["text"],
        "source": chunk["source"],
        "page": chunk["page"],
        "chunk_id": chunk["chunk_id"]
    })

with open(METADATA_FILE, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False)

print(f"Metadata saved to: {METADATA_FILE}")


# --------------------------------------------------
# 7. Finished
# --------------------------------------------------

print("\n" + "=" * 50)
print("VECTOR DATABASE CREATED SUCCESSFULLY")
print("=" * 50)

print(f"Vectors: {index.ntotal}")
print(f"Dimensions: {dimension}")
print(f"Index: {INDEX_FILE}")
print(f"Metadata: {METADATA_FILE}")