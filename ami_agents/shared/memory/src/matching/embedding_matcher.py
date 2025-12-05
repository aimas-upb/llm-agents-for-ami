"""Embedding Similarity Matcher (IM v1).

This module implements vector embedding-based intent matching
using sentence transformers for semantic similarity.
"""

import hashlib
import logging
import numpy as np
from typing import Any, Dict, List, Optional

from src.matching.base import IntentMatcher, MatchResult

logger = logging.getLogger(__name__)


class EmbeddingMatcher(IntentMatcher):
    """Intent Matcher v1 - Embedding Similarity.

    Uses sentence embeddings to compute semantic similarity between
    intent queries and signifiers.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_embeddings: bool = True):
        """Initialize the Embedding Matcher.

        Args:
            model_name: Sentence transformer model name
            cache_embeddings: Whether to cache signifier embeddings
        """
        super().__init__(version="v1")
        self.model_name = model_name
        self.cache_embeddings = cache_embeddings
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._model: Optional[Any] = None

    def _get_model(self):
        """Lazy load the sentence transformer model.

        Returns:
            Sentence transformer model

        Raises:
            ImportError: If sentence-transformers is not installed
        """
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info(f"Loading embedding model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
                logger.info("Embedding model loaded successfully")
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for EmbeddingMatcher. "
                    "Install it with: pip install sentence-transformers"
                )
        return self._model

    def match(
        self,
        intent_query: str,
        signifiers: List[Dict[str, Any]],
        k: int = 10,
        min_similarity: float = 0.0,
        **kwargs,
    ) -> List[MatchResult]:
        """Match intent query using embedding similarity.

        Args:
            intent_query: Natural language intent query
            signifiers: List of signifier dictionaries
            k: Number of top results to return
            min_similarity: Minimum similarity threshold (0.0 to 1.0)
            **kwargs: Additional parameters (ignored)

        Returns:
            List of MatchResult objects sorted by similarity

        Raises:
            ValueError: If inputs are invalid
        """
        if not intent_query:
            raise ValueError("intent_query cannot be empty")

        if not signifiers:
            logger.warning("No signifiers provided for matching")
            return []

        model = self._get_model()

        query_embedding = model.encode(intent_query, convert_to_numpy=True)

        results = []
        for signifier in signifiers:
            signifier_id = signifier.get("signifier_id", "unknown")

            signifier_embedding = self._get_signifier_embedding(
                signifier, model
            )

            similarity = self._cosine_similarity(
                query_embedding, signifier_embedding
            )

            if similarity >= min_similarity:
                results.append(
                    MatchResult(
                        signifier_id=signifier_id,
                        similarity=float(similarity),
                        metadata={
                            "matcher_version": self.version,
                            "model_name": self.model_name,
                            "embedding_dim": len(query_embedding),
                        },
                    )
                )

        results.sort(key=lambda x: x.similarity, reverse=True)

        logger.info(
            f"Embedding matching found {len(results)} matches "
            f"(min_similarity={min_similarity}), returning top {k}"
        )
        return results[:k]

    def _get_signifier_embedding(
        self, signifier: Dict[str, Any], model: Any
    ) -> np.ndarray:
        """Get or compute embedding for a signifier.

        Args:
            signifier: Signifier dictionary
            model: Sentence transformer model

        Returns:
            Embedding vector as numpy array
        """
        signifier_id = signifier.get("signifier_id", "unknown")

        if self.cache_embeddings:
            intent = signifier.get("intent", {})
            nl_text = intent.get("nl_text", "")

            cache_key = self._compute_cache_key(signifier_id, nl_text)

            if cache_key in self._embedding_cache:
                logger.debug(f"Using cached embedding for {signifier_id}")
                return self._embedding_cache[cache_key]

        text_to_encode = self._extract_signifier_text(signifier)

        embedding = model.encode(text_to_encode, convert_to_numpy=True)

        if self.cache_embeddings:
            self._embedding_cache[cache_key] = embedding
            logger.debug(f"Cached embedding for {signifier_id}")

        return embedding

    def _extract_signifier_text(self, signifier: Dict[str, Any]) -> str:
        """Extract text from signifier for embedding.

        Args:
            signifier: Signifier dictionary

        Returns:
            Combined text for embedding
        """
        intent = signifier.get("intent", {})
        nl_text = intent.get("nl_text", "")
        structured = intent.get("structured", {})

        text_parts = [nl_text]

        if structured and isinstance(structured, dict):
            intent_value = structured.get("intent", "")
            if intent_value:
                text_parts.append(intent_value)

        combined_text = " ".join(text_parts).strip()

        return combined_text if combined_text else "unknown intent"

    def _cosine_similarity(
        self, vec1: np.ndarray, vec2: np.ndarray
    ) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity (-1 to 1, normalized to 0 to 1)
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        cosine_sim = dot_product / (norm1 * norm2)

        normalized_sim = (cosine_sim + 1) / 2

        return float(np.clip(normalized_sim, 0.0, 1.0))

    def _compute_cache_key(self, signifier_id: str, text: str) -> str:
        """Compute cache key for embedding.

        Args:
            signifier_id: Signifier identifier
            text: Text content

        Returns:
            Cache key
        """
        content = f"{signifier_id}:{text}"
        return hashlib.md5(content.encode()).hexdigest()

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._embedding_cache.clear()
        logger.info("Embedding cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Cache statistics dictionary
        """
        return {
            "enabled": self.cache_embeddings,
            "size": len(self._embedding_cache),
            "model_name": self.model_name,
        }

    def get_info(self) -> Dict[str, Any]:
        """Get information about this matcher.

        Returns:
            Matcher information dictionary
        """
        return {
            "version": self.version,
            "name": "Embedding Similarity Matcher",
            "description": "Semantic similarity using sentence embeddings",
            "model": self.model_name,
            "parameters": {
                "min_similarity": {
                    "type": "float",
                    "default": 0.0,
                    "description": "Minimum similarity threshold (0.0 to 1.0)",
                }
            },
            "latency_budget_ms": 30,
        }
