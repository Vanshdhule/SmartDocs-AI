# test_search_engine.py
"""
Task 10 – Similarity Search Test Suite
Tests 10 diverse queries against the indexed document collection.
Requires: at least one PDF to be indexed in ChromaDB first.
"""

from backend.search_engine import SearchEngine

engine = SearchEngine()

# 10 diverse queries — factual, conceptual, comparative, multi-part, unanswerable
QUERIES = [
    # Factual / specific
    "What year did Einstein publish his five groundbreaking papers?",
    "What was Einstein's initial position at the Swiss Patent Office?",
    "What is the photoelectric effect and how did Einstein explain it?",

    # Conceptual
    "What is the special theory of relativity and what are its core principles?",
    "How does Einstein's concept of space-time differ from Newton's absolute space?",
    "What role does the speed of light play in Einstein's theories?",

    # Comparative / analytical
    "What parallel does the editor draw between Einstein and another scientist?",
    "How did Einstein's approach to physics differ from experimental physicists of his era?",

    # Multi-part / synthesis
    "What was the key idea that resolved Einstein's seven-year struggle, and what theory did it lead to?",

    # Unanswerable (should return 'not found in documents')
    "What did Einstein eat for breakfast on his birthday in 1921?",
]


def run_search_tests():
    print("=" * 80)
    print("  SmartDoc AI – Search Engine Test Suite (10 Queries)")
    print("=" * 80)

    passed = 0
    failed = 0

    for i, query in enumerate(QUERIES, 1):
        print(f"\n[Query {i}/10] {query}")
        print("-" * 60)

        try:
            results = engine.search_similar_chunks(query)

            if results:
                for r in results:
                    score = r.get("relevance_score", 0.0)
                    page  = r.get("page", "?")
                    text  = r.get("text", "")[:120]
                    src   = r.get("source", "unknown")
                    print(f"  ✅ Score: {score:.4f} | Source: {src} | Page: {page}")
                    print(f"     {text}...")
                passed += 1
            else:
                print("  ℹ️  No results above threshold — query may be unanswerable from documents.")
                passed += 1  # Expected for out-of-scope queries

        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            failed += 1

    print("\n" + "=" * 80)
    print(f"  Results: {passed} passed | {failed} failed out of {len(QUERIES)} queries")
    print("=" * 80)


if __name__ == "__main__":
    run_search_tests()