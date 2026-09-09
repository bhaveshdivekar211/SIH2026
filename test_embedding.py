import ollama

response = ollama.embed(
    model="bge-m3:latest",
    input="This is a test sentence for my RAG chatbot."
)

embedding = response["embeddings"][0]

print("Embedding dimensions:", len(embedding))
print("First 10 values:", embedding[:10])