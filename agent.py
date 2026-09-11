"""
agent.py
Run this file to chat with the already-built vector database.

Make sure you've run `python ingest.py` at least once before this
(it needs faiss_index/ to exist).

Usage:  python agent.py
"""

import os
import sys
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from config import FAISS_INDEX_DIR, EMBEDDING_MODEL, CHAT_MODEL, RETRIEVAL_K

load_dotenv()


def load_vector_store():
    if not FAISS_INDEX_DIR.exists():
        sys.exit(
            f"No index found at '{FAISS_INDEX_DIR}/'.\n"
            "Run 'python ingest.py' first to build the database."
        )
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return FAISS.load_local(
        str(FAISS_INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


vector_store = load_vector_store()


@tool
def search_docs(query: str) -> str:
    """Search the ingested documents and return the most relevant excerpts."""
    results = vector_store.similarity_search(query, k=RETRIEVAL_K)
    if not results:
        return "No relevant information found in the documents."

    formatted = []
    for doc in results:
        source = doc.metadata.get("source_file", "unknown source")
        page = doc.metadata.get("page", "?")
        formatted.append(f"[{source}, page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def build_agent():
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        sys.exit("NVIDIA_API_KEY not set. Add it to your .env file.")

    model = ChatNVIDIA(model=CHAT_MODEL, api_key=api_key)

    return create_agent(
        model=model,
        tools=[search_docs],
        system_prompt=(
            "You are a helpful assistant who answers questions accurately "
            "by referring to the provided documents. If the answer isn't in "
            "the documents, say so instead of guessing."
        ),
    )


def main():
    agent = build_agent()
    print("Agent ready. Type your question (or 'exit' to quit).\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue

        try:
            response = agent.invoke({"messages": [{"role": "user", "content": query}]})
            print("\nAgent:", response["messages"][-1].content, "\n")
        except Exception as e:
            print(f"\nError getting response: {e}\n")


if __name__ == "__main__":
    main()
