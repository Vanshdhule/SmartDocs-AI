# tests/test_nvidia.py
# Run with: python tests/test_nvidia.py
import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

# Force UTF-8 output so emoji print correctly on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

chat_key      = os.getenv("NVIDIA_CHAT_API_KEY")
embedding_key = os.getenv("NVIDIA_EMBEDDING_API_KEY")
base_url      = "https://integrate.api.nvidia.com/v1"

if not chat_key:
    print("FAIL: NVIDIA_CHAT_API_KEY not found in .env")
    sys.exit(1)
if not embedding_key:
    print("FAIL: NVIDIA_EMBEDDING_API_KEY not found in .env")
    sys.exit(1)

# ── Test 1: Chat model ────────────────────────────────────────────────────────
print("Testing chat model (meta/llama-3.3-70b-instruct)...")
try:
    chat_client = OpenAI(api_key=chat_key, base_url=base_url)
    resp = chat_client.chat.completions.create(
        model="meta/llama-3.3-70b-instruct",
        messages=[{"role": "user", "content": "Reply with just: OK"}],
        max_tokens=5,
    )
    print(f"  OK  Chat model working: {resp.choices[0].message.content.strip()}")
except Exception as e:
    print(f"  FAIL Chat model failed: {e}")

# ── Test 2: Embedding model ───────────────────────────────────────────────────
print("Testing embedding model (nvidia/nv-embedqa-e5-v5)...")
try:
    embed_client = OpenAI(api_key=embedding_key, base_url=base_url)
    resp = embed_client.embeddings.create(
        model="nvidia/nv-embedqa-e5-v5",
        input="test sentence for embedding",
        extra_body={"input_type": "passage", "truncate": "END"},
    )
    dim = len(resp.data[0].embedding)
    print(f"  OK  Embedding model working: dimension = {dim}")
    if dim != 1024:
        print(f"  WARN Expected 1024 dims, got {dim} -- update EMBEDDING_DIMENSION in config.py")
except Exception as e:
    print(f"  FAIL Embedding model failed: {e}")

print("\nDone.")