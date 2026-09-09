from pypdf import PdfReader
import os
import json


def extract_pages(pdf_path):
    reader = PdfReader(pdf_path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text:
            pages.append({
                "page": page_number,
                "text": text
            })

    return pages


def chunk_text(text, chunk_size=1000, overlap=200):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# --------------------------------
# PROCESS ALL PDFS
# --------------------------------

pdf_folder = "pdfs"

all_chunks = []

for filename in os.listdir(pdf_folder):

    if filename.lower().endswith(".pdf"):

        pdf_path = os.path.join(pdf_folder, filename)

        print(f"Processing: {filename}")

        pages = extract_pages(pdf_path)

        for page in pages:

            chunks = chunk_text(page["text"])

            for chunk_number, chunk in enumerate(chunks, start=1):

                all_chunks.append({
                    "text": chunk,
                    "source": filename,
                    "page": page["page"],
                    "chunk_id": chunk_number
                })


# --------------------------------
# PROCESSING COMPLETE
# --------------------------------

print("\n==============================")
print("PROCESSING COMPLETE")
print("==============================")

print(f"Total chunks: {len(all_chunks)}")


# --------------------------------
# SAVE CHUNKS TO JSON
# --------------------------------

with open("chunks.json", "w", encoding="utf-8") as file:

    json.dump(
        all_chunks,
        file,
        ensure_ascii=False,
        indent=2
    )

print("Chunks saved to chunks.json")