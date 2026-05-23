import time
import json
from pathlib import Path

from backend.pdf_processor import PDFProcessor
from backend.text_chunker import TextChunker
from backend.embeddings import EmbeddingGenerator


# ==========================
# CONFIG
# ==========================
PDF_PATH = "uploads/Albert_Einstein,_John_Stachel,_Roger_Penrose_Einstein's_Miraculous.pdf"  # <-- put your real PDF path here
OUTPUT_FILE = "pdf_embeddings_output.json"
CHUNK_STRATEGY = "sentences"


def test_pdf_to_embeddings():
    start_time = time.time()

    pdf_path = Path(PDF_PATH)
    if not pdf_path.exists():
        raise FileNotFoundError(f"❌ PDF not found: {pdf_path}")

    print("📄 Extracting text from PDF...")

    pdf_processor = PDFProcessor()
    with open(pdf_path, "rb") as f:
        pages = pdf_processor.extract_text(f)

    print(f"✅ Extracted {len(pages)} pages")

    # ==========================
    # CHUNKING
    # ==========================
    print("✂️ Chunking text...")

    chunker = TextChunker()
    chunks = []

    for page in pages:
        if not page["text"].strip():
            continue

        page_chunks = chunker.create_chunks(
            text=page["text"],
            source_file=pdf_path.name,
            page_number=page["page_number"],
            strategy=CHUNK_STRATEGY
        )
        chunks.extend(page_chunks)

    print(f"✅ Created {len(chunks)} text chunks")

    if not chunks:
        raise RuntimeError("❌ No chunks generated from PDF")

    # ==========================
    # EMBEDDINGS
    # ==========================
    print("🧠 Generating embeddings...")

    generator = EmbeddingGenerator()
    embedding_data = generator.prepare_embedding_data(chunks)

    # ==========================
    # SAVE OUTPUT
    # ==========================
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(embedding_data, f, indent=2)

    elapsed = time.time() - start_time

    # ==========================
    # SUMMARY
    # ==========================
    print("\n🎉 PDF → Embeddings Pipeline Complete")
    print(f"📦 Total embeddings: {len(embedding_data)}")
    print(f"📐 Embedding dimension: {len(embedding_data[0]['embedding'])}")
    print(f"⏱ Processing time: {elapsed:.2f} seconds")
    print(f"💾 Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    test_pdf_to_embeddings()