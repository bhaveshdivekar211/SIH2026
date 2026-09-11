import json
import hashlib
import time
from pathlib import Path

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

from config import (
    DOCS_DIR,
    FAISS_INDEX_DIR,
    MANIFEST_PATH,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

load_dotenv()

def file_hash(path: Path) -> str:

    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)

    return h.hexdigest()


def load_manifest() -> dict:

    if MANIFEST_PATH.exists():

        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    return {}


def save_manifest(manifest: dict) -> None:

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:

        json.dump(
            manifest,
            f,
            ensure_ascii=False,
            indent=2
        )


def find_new_or_changed_pdfs(manifest: dict) -> list[Path]:

    if not DOCS_DIR.exists():

        raise FileNotFoundError(
            f"'{DOCS_DIR}/' folder not found. "
            f"Create it and put your PDFs inside."
        )

    pdf_paths = sorted(DOCS_DIR.glob("*.pdf"))

    if not pdf_paths:

        raise FileNotFoundError(
            f"No PDF files found in '{DOCS_DIR}/'."
        )

    to_process = []

    for path in pdf_paths:

        current_hash = file_hash(path)

        record = manifest.get(path.name)

        # New PDF
        if record is None:

            to_process.append(path)

        # Existing PDF but contents changed
        elif record["hash"] != current_hash:

            to_process.append(path)

    return to_process


def chunk_pdf(path: Path) -> list[Document]:

    print(f"\nReading PDF: {path.name}")

    try:

        loader = PyPDFLoader(str(path))

        documents = loader.load()

    except Exception as e:

        print(
            f"  Error reading '{path.name}': {e}"
        )

        return []

    if not documents:

        print(
            f"  Warning: '{path.name}' produced no readable text."
        )

        print(
            "  This may be a scanned/image-only PDF."
        )

        return []

    print(
        f"  Pages loaded: {len(documents)}"
    )


    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(documents)


    for chunk in chunks:

        chunk.metadata["source_file"] = path.name

    print(
        f"  Chunks created: {len(chunks)}"
    )

    return chunks


def add_chunks_in_batches(
    vector_store,
    chunks,
    embeddings,
    batch_size=16
):
    """
    Embed documents in small batches and add them to FAISS.

    Batching prevents Ollama's model runner from receiving
    too many documents at once and crashing.
    """

    total = len(chunks)

    if total == 0:

        return vector_store

    for i in range(0, total, batch_size):

        batch = chunks[
            i:i + batch_size
        ]

        end = min(
            i + batch_size,
            total
        )

        print(
            f"\nEmbedding chunks "
            f"{i + 1}-{end} of {total}..."
        )

        try:

            # First batch creates the FAISS index
            if vector_store is None:

                vector_store = FAISS.from_documents(
                    batch,
                    embeddings
                )

            # Remaining batches are appended
            else:

                vector_store.add_documents(
                    batch
                )

            print(
                f"  Successfully embedded {len(batch)} chunks."
            )

        except Exception as e:

            print(
                "\nERROR while embedding batch:"
            )

            print(e)

            print(
                "\nThe Ollama model runner may have crashed."
            )

            print(
                "Try reducing BATCH_SIZE from 16 to 8."
            )

            raise

        # Small pause gives Ollama time to recover
        time.sleep(0.5)

    return vector_store


def main():

    print("PDF INGESTION STARTED")
    manifest = load_manifest()


    pdf_paths = find_new_or_changed_pdfs(
        manifest
    )

    if not pdf_paths:

        print(
            "\nNo new or changed PDFs found."
        )

        print(
            "Your FAISS vector database is already up to date."
        )

        return


    print(
        f"\nFound {len(pdf_paths)} new/changed PDF(s):"
    )

    for path in pdf_paths:

        print(
            f"  - {path.name}"
        )


    all_new_chunks = []

    for path in pdf_paths:

        chunks = chunk_pdf(path)

        all_new_chunks.extend(
            chunks
        )


    if not all_new_chunks:

        print(
            "\nNo chunks were created."
        )

        return



    print(
        f"TOTAL NEW CHUNKS: {len(all_new_chunks)}"
    )

    print(
        f"\nInitialising embedding model:"
    )

    print(
        f"  {EMBEDDING_MODEL}"
    )

    try:

        embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url="http://127.0.0.1:11434"
        )

    except Exception as e:

        print(
            "\nError initialising embeddings."
        )

        print(
            "Make sure Ollama is running."
        )

        print(e)

        return


    if FAISS_INDEX_DIR.exists():

        print(
            "\nExisting FAISS index found."
        )

        print(
            "Loading existing index..."
        )

        vector_store = FAISS.load_local(
            str(FAISS_INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    else:

        print(
            "\nNo existing FAISS index found."
        )

        print(
            "A new index will be created."
        )

        vector_store = None


    BATCH_SIZE = 16

    print(
        f"\nEmbedding using batch size: {BATCH_SIZE}"
    )

    vector_store = add_chunks_in_batches(
        vector_store=vector_store,
        chunks=all_new_chunks,
        embeddings=embeddings,
        batch_size=BATCH_SIZE
    )


    print(
        "\nSaving FAISS vector database..."
    )

    vector_store.save_local(
        str(FAISS_INDEX_DIR)
    )


    for path in pdf_paths:

        manifest[path.name] = {
            "hash": file_hash(path)
        }


    save_manifest(
        manifest
    )


    print(
        "INGESTION COMPLETE"
    )

    print(
        f"Files processed: {len(pdf_paths)}"
    )

    print(
        f"New chunks added: {len(all_new_chunks)}"
    )

    print(
        f"Total files tracked: {len(manifest)}"
    )

    print(
        f"FAISS index: {FAISS_INDEX_DIR}/"
    )

if __name__ == "__main__":

    main()