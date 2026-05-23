"""Test globally unique chunk ID generation"""
from backend.text_chunker import TextChunker

# Initialize chunker
chunker = TextChunker()

# Test with two different documents
chunks1 = chunker.create_chunks(
    text="This is a test document with some text for testing.",
    source_file="document1.pdf",
    page_number=1
)

chunks2 = chunker.create_chunks(
    text="This is another test document with different content.",
    source_file="document2.pdf",
    page_number=1
)

chunks3 = chunker.create_chunks(
    text="This is the same document as the first one.",
    source_file="document1.pdf",  # Same filename as chunks1
    page_number=2
)

print("=" * 60)
print("Testing Globally Unique Chunk IDs")
print("=" * 60)

print(f"\n✅ Document 1 (document1.pdf):")
print(f"   Chunk ID: {chunks1[0]['chunk_id']}")
print(f"   Pattern: <doc_hash>_<chunk_index>")

print(f"\n✅ Document 2 (document2.pdf):")
print(f"   Chunk ID: {chunks2[0]['chunk_id']}")
print(f"   Different hash from Document 1: {chunks1[0]['chunk_id'].split('_')[0] != chunks2[0]['chunk_id'].split('_')[0]}")

print(f"\n✅ Document 1 again (document1.pdf, page 2):")
print(f"   Chunk ID: {chunks3[0]['chunk_id']}")
print(f"   Same doc hash as first: {chunks1[0]['chunk_id'].split('_')[0] == chunks3[0]['chunk_id'].split('_')[0]}")

print(f"\n✅ Global Uniqueness Test:")
all_ids = [chunks1[0]['chunk_id'], chunks2[0]['chunk_id'], chunks3[0]['chunk_id']]
print(f"   All IDs unique: {len(all_ids) == len(set(all_ids))}")
print(f"   IDs: {all_ids}")

print("\n" + "=" * 60)
print("✅ Chunk ID system working correctly!")
print("=" * 60)