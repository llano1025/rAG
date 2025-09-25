"""
Maximal Marginal Relevance (MMR) Diversifier for search result diversification.

Implements the classic MMR algorithm to reduce redundancy and improve coverage
in search results by balancing relevance and diversity.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import asyncio
from concurrent.futures import ThreadPoolExecutor

from api.schemas.search_schemas import SearchResult

logger = logging.getLogger(__name__)


class MMRDiversifier:
    """
    Maximal Marginal Relevance (MMR) implementation for result diversification.

    MMR Score = λ * Relevance(q, d) - (1-λ) * max Similarity(d, selected)

    Where:
    - λ (lambda_param): controls trade-off between relevance and diversity
    - Higher λ: favor relevance over diversity
    - Lower λ: favor diversity over relevance
    """

    def __init__(
        self,
        lambda_param: float = 0.7,
        similarity_threshold: float = 0.8,
        max_results: Optional[int] = None,
        similarity_metric: str = "cosine"
    ):
        """
        Initialize MMR diversifier.

        Args:
            lambda_param: Balance between relevance (1.0) and diversity (0.0)
            similarity_threshold: Minimum similarity to apply diversity penalty
            max_results: Maximum number of diversified results to return
            similarity_metric: Similarity metric ("cosine", "euclidean", "dot_product")
        """
        if not 0.0 <= lambda_param <= 1.0:
            raise ValueError("lambda_param must be between 0.0 and 1.0")

        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")

        self.lambda_param = lambda_param
        self.similarity_threshold = similarity_threshold
        self.max_results = max_results
        self.similarity_metric = similarity_metric
        self._executor = ThreadPoolExecutor(max_workers=4)

        logger.debug(
            f"Initialized MMR diversifier: λ={lambda_param}, "
            f"threshold={similarity_threshold}, metric={similarity_metric}"
        )

    async def diversify(
        self,
        results: List[Union[SearchResult, Dict[str, Any]]],
        lambda_param: Optional[float] = None,
        max_results: Optional[int] = None
    ) -> List[Union[SearchResult, Dict[str, Any]]]:
        """
        Apply MMR diversification to search results.

        Args:
            results: List of search results with embeddings
            lambda_param: Override default lambda parameter
            max_results: Override default max results

        Returns:
            Diversified list of results
        """
        if not results:
            return results

        # Use provided parameters or defaults
        lambda_val = lambda_param if lambda_param is not None else self.lambda_param
        max_res = max_results if max_results is not None else self.max_results

        if max_res is None:
            max_res = len(results)
        else:
            max_res = min(max_res, len(results))

        logger.debug(f"Starting MMR diversification: {len(results)} → {max_res} results, λ={lambda_val}")

        try:
            # Extract embeddings and validate
            embeddings = await self._extract_embeddings(results)
            if not embeddings:
                logger.warning("No embeddings found in results, returning original results")
                return results[:max_res]

            # Perform MMR selection
            diversified_indices = await self._mmr_select(
                results, embeddings, lambda_val, max_res
            )

            # Return diversified results
            diversified_results = [results[i] for i in diversified_indices]

            logger.debug(
                f"MMR diversification completed: {len(results)} → {len(diversified_results)} results"
            )

            return diversified_results

        except Exception as e:
            logger.error(f"MMR diversification failed: {e}")
            # Fallback: return top results without diversification
            return results[:max_res]

    async def _extract_embeddings(
        self,
        results: List[Union[SearchResult, Dict[str, Any]]]
    ) -> List[Optional[np.ndarray]]:
        """Extract embeddings from search results."""
        embeddings = []

        for result in results:
            embedding = None

            if isinstance(result, SearchResult):
                # Try to get embedding from metadata
                if hasattr(result, 'embedding') and result.embedding:
                    embedding = np.array(result.embedding)
                elif 'embedding' in result.metadata:
                    embedding = np.array(result.metadata['embedding'])
            elif isinstance(result, dict):
                # Try various embedding fields
                if 'embedding' in result:
                    embedding = np.array(result['embedding'])
                elif 'metadata' in result and 'embedding' in result['metadata']:
                    embedding = np.array(result['metadata']['embedding'])
                elif 'vector' in result:
                    embedding = np.array(result['vector'])

            embeddings.append(embedding)

        return embeddings

    async def _mmr_select(
        self,
        results: List[Union[SearchResult, Dict[str, Any]]],
        embeddings: List[Optional[np.ndarray]],
        lambda_param: float,
        max_results: int
    ) -> List[int]:
        """
        Perform MMR selection algorithm.

        Returns:
            List of indices of selected results
        """
        # Filter results with valid embeddings
        valid_indices = [
            i for i, emb in enumerate(embeddings)
            if emb is not None and len(emb) > 0
        ]

        if not valid_indices:
            logger.warning("No valid embeddings found, returning top results by score")
            return list(range(min(max_results, len(results))))

        # Get valid embeddings and scores
        valid_embeddings = [embeddings[i] for i in valid_indices]
        valid_scores = []

        for i in valid_indices:
            result = results[i]
            if isinstance(result, SearchResult):
                score = result.score
            elif isinstance(result, dict):
                score = result.get('score', 0.0)
            else:
                score = 0.0
            valid_scores.append(score)

        # Normalize scores to 0-1 range
        if valid_scores:
            min_score = min(valid_scores)
            max_score = max(valid_scores)
            if max_score > min_score:
                normalized_scores = [
                    (score - min_score) / (max_score - min_score)
                    for score in valid_scores
                ]
            else:
                normalized_scores = [1.0] * len(valid_scores)
        else:
            normalized_scores = [0.0] * len(valid_indices)

        # Run MMR selection in executor to avoid blocking
        loop = asyncio.get_event_loop()
        selected_indices = await loop.run_in_executor(
            self._executor,
            self._mmr_greedy_selection,
            valid_indices,
            valid_embeddings,
            normalized_scores,
            lambda_param,
            min(max_results, len(valid_indices))
        )

        return selected_indices

    def _mmr_greedy_selection(
        self,
        valid_indices: List[int],
        embeddings: List[np.ndarray],
        scores: List[float],
        lambda_param: float,
        max_results: int
    ) -> List[int]:
        """
        Greedy MMR selection algorithm.

        Args:
            valid_indices: Original indices of results with valid embeddings
            embeddings: Valid embeddings corresponding to valid_indices
            scores: Normalized relevance scores
            lambda_param: Relevance vs diversity trade-off
            max_results: Maximum results to select

        Returns:
            List of original indices of selected results
        """
        selected = []
        remaining = list(range(len(valid_indices)))

        # Convert embeddings to matrix for efficient similarity computation
        embedding_matrix = np.vstack(embeddings)

        while len(selected) < max_results and remaining:
            best_score = -float('inf')
            best_idx = 0

            for i, candidate_idx in enumerate(remaining):
                # Get relevance score
                relevance_score = scores[candidate_idx]

                if selected:
                    # Compute similarities to already selected documents
                    candidate_embedding = embedding_matrix[candidate_idx].reshape(1, -1)
                    selected_embeddings = embedding_matrix[selected]

                    # Compute similarity based on metric
                    if self.similarity_metric == "cosine":
                        similarities = cosine_similarity(candidate_embedding, selected_embeddings)[0]
                    elif self.similarity_metric == "dot_product":
                        similarities = np.dot(candidate_embedding, selected_embeddings.T)[0]
                    elif self.similarity_metric == "euclidean":
                        # Convert to similarity (inverse of distance)
                        distances = np.linalg.norm(
                            candidate_embedding - selected_embeddings, axis=1
                        )
                        similarities = 1.0 / (1.0 + distances)
                    else:
                        similarities = cosine_similarity(candidate_embedding, selected_embeddings)[0]

                    # Apply similarity threshold
                    max_similarity = np.max(similarities)
                    if max_similarity < self.similarity_threshold:
                        max_similarity = 0.0  # No diversity penalty if below threshold
                else:
                    max_similarity = 0.0

                # Compute MMR score
                mmr_score = lambda_param * relevance_score - (1 - lambda_param) * max_similarity

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            # Select the best candidate
            selected_candidate_idx = remaining.pop(best_idx)
            selected.append(selected_candidate_idx)

        # Convert back to original indices
        return [valid_indices[i] for i in selected]

    def compute_similarity_matrix(
        self,
        embeddings: List[np.ndarray]
    ) -> np.ndarray:
        """
        Compute similarity matrix for a set of embeddings.

        Args:
            embeddings: List of embedding vectors

        Returns:
            Similarity matrix
        """
        if not embeddings:
            return np.array([])

        embedding_matrix = np.vstack(embeddings)

        if self.similarity_metric == "cosine":
            return cosine_similarity(embedding_matrix)
        elif self.similarity_metric == "dot_product":
            return np.dot(embedding_matrix, embedding_matrix.T)
        elif self.similarity_metric == "euclidean":
            # Convert distances to similarities
            from sklearn.metrics.pairwise import euclidean_distances
            distances = euclidean_distances(embedding_matrix)
            return 1.0 / (1.0 + distances)
        else:
            return cosine_similarity(embedding_matrix)

    def analyze_diversity(
        self,
        results: List[Union[SearchResult, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Analyze diversity metrics of a result set.

        Args:
            results: List of search results

        Returns:
            Dictionary with diversity metrics
        """
        try:
            if not results:
                return {"error": "No results provided"}

            # Extract text content for diversity analysis
            texts = []
            for result in results:
                if isinstance(result, SearchResult):
                    texts.append(result.text)
                elif isinstance(result, dict):
                    texts.append(result.get('text', ''))

            # Basic diversity metrics
            unique_documents = len(set(
                result.document_id if isinstance(result, SearchResult)
                else result.get('document_id')
                for result in results
            ))

            # Text similarity analysis would go here
            # For now, provide basic metrics

            return {
                "total_results": len(results),
                "unique_documents": unique_documents,
                "document_diversity_ratio": unique_documents / len(results) if results else 0.0,
                "avg_text_length": np.mean([len(text) for text in texts]) if texts else 0,
                "text_length_variance": np.var([len(text) for text in texts]) if texts else 0
            }

        except Exception as e:
            logger.error(f"Diversity analysis failed: {e}")
            return {"error": str(e)}

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._executor:
            self._executor.shutdown(wait=True)

    def __del__(self):
        """Cleanup executor on deletion."""
        if hasattr(self, '_executor') and self._executor:
            self._executor.shutdown(wait=False)


# Factory functions for common MMR configurations
def create_relevance_focused_mmr(similarity_threshold: float = 0.9) -> MMRDiversifier:
    """Create MMR diversifier focused on relevance (λ=0.8)."""
    return MMRDiversifier(
        lambda_param=0.8,
        similarity_threshold=similarity_threshold
    )

def create_diversity_focused_mmr(similarity_threshold: float = 0.7) -> MMRDiversifier:
    """Create MMR diversifier focused on diversity (λ=0.3)."""
    return MMRDiversifier(
        lambda_param=0.3,
        similarity_threshold=similarity_threshold
    )

def create_balanced_mmr(similarity_threshold: float = 0.8) -> MMRDiversifier:
    """Create balanced MMR diversifier (λ=0.6)."""
    return MMRDiversifier(
        lambda_param=0.6,
        similarity_threshold=similarity_threshold
    )


# Singleton manager for default MMR instance
_default_mmr_diversifier = None

def get_default_mmr_diversifier() -> MMRDiversifier:
    """Get or create the default MMR diversifier instance."""
    global _default_mmr_diversifier

    if _default_mmr_diversifier is None:
        _default_mmr_diversifier = create_balanced_mmr()

    return _default_mmr_diversifier

def set_default_mmr_config(
    lambda_param: float = 0.6,
    similarity_threshold: float = 0.8,
    max_results: Optional[int] = None,
    similarity_metric: str = "cosine"
):
    """Configure the default MMR diversifier."""
    global _default_mmr_diversifier

    _default_mmr_diversifier = MMRDiversifier(
        lambda_param=lambda_param,
        similarity_threshold=similarity_threshold,
        max_results=max_results,
        similarity_metric=similarity_metric
    )