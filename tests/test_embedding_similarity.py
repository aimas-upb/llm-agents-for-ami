"""
Test script for sentence embedding similarity.

This script demonstrates semantic similarity between implicit intent phrases
using the sentence-transformers library with the all-MiniLM-L6-v2 model.

Run with:
    conda run -n ami-agents python tests/test_embedding_similarity.py
"""

import sys


def test_embedding_similarity():
    """Test cosine similarity between implicit intent phrases."""
    try:
        from sentence_transformers import SentenceTransformer, util
    except ImportError:
        print("ERROR: sentence-transformers not installed")
        print("Install with: pip install sentence-transformers")
        return False

    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        return False

    sentences = [
        "It's too dark in here",
        "I can't see anything on my desk",
    ]

    try:
        embeddings = model.encode(sentences, convert_to_tensor=True)
        cosine_sim = util.cos_sim(embeddings[0], embeddings[1]).item()

        print(f"\nSentence 1: {sentences[0]}")
        print(f"Sentence 2: {sentences[1]}")
        print(f"Cosine similarity: {cosine_sim:.4f}")
        print()

        # Additional test cases for implicit vs explicit intents
        print("Additional test cases:")
        print("-" * 60)

        test_cases = [
            ("turn on the light", "activate the lamp", "Explicit intent (same action, different phrasing)"),
            ("it's too dark", "the room is dim", "Implicit intent (both describe darkness)"),
            ("set brightness to 75", "increase brightness to 75", "Explicit intent (similar action)"),
            ("I'm cold", "it's chilly in here", "Implicit intent (same condition)"),
            ("turn on the light", "make it darker", "Conflicting intents"),
        ]

        for sent1, sent2, description in test_cases:
            embs = model.encode([sent1, sent2], convert_to_tensor=True)
            sim = util.cos_sim(embs[0], embs[1]).item()
            print(f"{description}")
            print(f"  '{sent1}' vs '{sent2}'")
            print(f"  Similarity: {sim:.4f}\n")

        return True

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return False


if __name__ == "__main__":
    success = test_embedding_similarity()
    sys.exit(0 if success else 1)
