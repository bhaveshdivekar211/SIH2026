"""Shared configuration for ingest.py and agent.py"""

from pathlib import Path

# Folder where you drop PDFs to be ingested
DOCS_DIR = Path("docs")

# Where the FAISS vector index is stored on disk
FAISS_INDEX_DIR = Path("faiss_index")

# Tracks which files (and versions of files) have already been embedded,
# so ingest.py can skip/append instead of rebuilding everything from scratch
MANIFEST_PATH = Path("manifest.json")

# Embedding model served by Ollama (must be pulled locally: `ollama pull bge-m3`)
EMBEDDING_MODEL = "bge-m3"

# Chat model served via NVIDIA NIM
CHAT_MODEL = "nvidia/nemotron-3-super-120b-a12b"

# Chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# How many chunks to retrieve per query
RETRIEVAL_K = 3
