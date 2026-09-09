import json
import ollama
import time

INPUT_FILE = "chunks.json"
OUTPUT_FILE = "chunks_with_embeddings.json"

BATCH_SIZE = 32
MODEL = "bge-m3:latest"


# Load chunks
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Total chunks: {len(chunks)}")
print(f"Embedding model: {MODEL}")
print("-" * 50)


# Process in batches
for start in range(0, len(chunks), BATCH_SIZE):

    end = min(start + BATCH_SIZE, len(chunks))
    batch = chunks[start:end]

    texts = [chunk["text"] for chunk in batch]

    print(f"Embedding chunks {start + 1} to {end}...")

    try:
        response = ollama.embed(
            model=MODEL,
            input=texts
        )

        embeddings = response["embeddings"]

        # Add embedding to each chunk
        for chunk, embedding in zip(batch, embeddings):
            chunk["embedding"] = embedding

        print(f"Completed: {end}/{len(chunks)}")

    except Exception as e:
        print(f"ERROR in batch {start + 1}-{end}: {e}")
        print("Stopping.")
        break


# Save result
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(chunks, f)

print("-" * 50)
print("PROCESSING COMPLETE")
print(f"Saved to: {OUTPUT_FILE}")