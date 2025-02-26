# llm/context_optimizer.py

from typing import List, Dict, Optional
import numpy as np
from dataclasses import dataclass
import logging

class ContextError(Exception):
    """Raised when context optimization fails."""
    pass

@dataclass
class ContextWindow:
    """Represents an optimized context window."""
    text: str
    token_count: int
    relevance_score: float
    metadata: Dict

class ContextOptimizer:
    """Optimizes context window composition for LLM queries."""
    
    def __init__(
        self,
        max_tokens: int,
        tokenizer: callable,
        similarity_threshold: float = 0.7
    ):
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer
        self.similarity_threshold = similarity_threshold

    def optimize_context(
        self,
        query: str,
        context_chunks: List[Dict],
        query_embedding: Optional[np.ndarray] = None,
        previous_responses: Optional[List[str]] = None
    ) -> ContextWindow:
        """Create optimized context window from available chunks."""
        try:
            # Calculate relevance scores
            scored_chunks = self._score_chunks(query, context_chunks, query_embedding)
            
            # Filter relevant chunks
            relevant_chunks = [
                chunk for chunk in scored_chunks
                if chunk["relevance_score"] >= self.similarity_threshold
            ]
            
            # Sort by relevance
            relevant_chunks.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            # Build context window
            context_text = []
            current_tokens = 0
            
            # Add previous responses if provided
            if previous_responses:
                for response in previous_responses[-2:]:  # Add last 2 responses
                    response_tokens = len(self.tokenizer(response))
                    if current_tokens + response_tokens <= self.max_tokens:
                        context_text.append(response)
                        current_tokens += response_tokens
            
            # Add relevant chunks
            for chunk in relevant_chunks:
                chunk_tokens = len(self.tokenizer(chunk["text"]))
                if current_tokens + chunk_tokens <= self.max_tokens:
                    context_text.append(chunk["text"])
                    current_tokens += chunk_tokens
                else:
                    break
            
            return ContextWindow(
                text="\n".join(context_text),
                token_count=current_tokens,
                relevance_score=np.mean([chunk["relevance_score"] for chunk in relevant_chunks]) if relevant_chunks else 0.0,
                metadata={"num_chunks": len(context_text)}
            )
        except Exception as e:
            logging.error(f"Context optimization failed: {str(e)}")
            raise ContextError(f"Failed to optimize context: {str(e)}")

    def _score_chunks(
        self,
        query: str,
        chunks: List[Dict],
        query_embedding: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """Score chunks based on relevance to query."""
        try:
            scored_chunks = []
            
            for chunk in chunks:
                if query_embedding is not None and "embedding" in chunk:
                    # Use vector similarity if embeddings available
                    relevance_score = self._calculate_similarity(
                        query_embedding,
                        chunk["embedding"]
                    )
                else:
                    # Fallback to simple keyword matching
                    relevance_score = self._keyword_relevance(query, chunk["text"])
                
                scored_chunks.append({
                    **chunk,
                    "relevance_score": float(relevance_score)
                })
            
            return scored_chunks
        except Exception as e:
            logging.error(f"Chunk scoring failed: {str(e)}")
            raise ContextError(f"Failed to score chunks: {str(e)}")

    def _calculate_similarity(
        self,
        query_embedding: np.ndarray,
        chunk_embedding: np.ndarray
    ) -> float:
        """Calculate cosine similarity between embeddings."""
        try:
            # Normalize embeddings
            query_norm = np.linalg.norm(query_embedding)
            chunk_norm = np.linalg.norm(chunk_embedding)
            
            if query_norm == 0 or chunk_norm == 0:
                return 0.0
                
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, chunk_embedding) / (query_norm * chunk_norm)
            return float(similarity)
        except Exception as e:
            logging.error(f"Similarity calculation failed: {str(e)}")
            return 0.0

    def _keyword_relevance(self, query: str, text: str) -> float:
        """Calculate keyword-based relevance score."""
        try:
            # Normalize and tokenize text
            query_words = set(query.lower().split())
            text_words = set(text.lower().split())
            
            if not query_words:
                return 0.0
            
            # Calculate word overlap ratio
            overlap = len(query_words.intersection(text_words))
            relevance = overlap / len(query_words)
            
            return float(relevance)
        except Exception as e:
            logging.error(f"Keyword relevance calculation failed: {str(e)}")
            return 0.0

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        try:
            return len(self.tokenizer(text))
        except Exception as e:
            logging.error(f"Token estimation failed: {str(e)}")
            raise ContextError(f"Failed to estimate tokens: {str(e)}")

    def merge_contexts(
        self,
        contexts: List[ContextWindow],
        max_contexts: int = 5
    ) -> ContextWindow:
        """Merge multiple context windows while respecting token limit."""
        try:
            # Sort contexts by relevance score
            sorted_contexts = sorted(
                contexts,
                key=lambda x: x.relevance_score,
                reverse=True
            )[:max_contexts]
            
            # Combine texts
            merged_text = []
            current_tokens = 0
            
            for context in sorted_contexts:
                if current_tokens + context.token_count <= self.max_tokens:
                    merged_text.append(context.text)
                    current_tokens += context.token_count
                else:
                    break
            
            # Calculate average relevance score
            avg_relevance = np.mean([c.relevance_score for c in sorted_contexts])
            
            return ContextWindow(
                text="\n\n".join(merged_text),
                token_count=current_tokens,
                relevance_score=avg_relevance,
                metadata={
                    "num_contexts": len(merged_text),
                    "original_contexts": len(contexts)
                }
            )
        except Exception as e:
            logging.error(f"Context merging failed: {str(e)}")
            raise ContextError(f"Failed to merge contexts: {str(e)}")

    def validate_context(self, context: ContextWindow) -> bool:
        """Validate a context window."""
        try:
            if not context.text.strip():
                return False
                
            actual_tokens = self.estimate_tokens(context.text)
            if actual_tokens > self.max_tokens:
                return False
                
            if context.relevance_score < 0 or context.relevance_score > 1:
                return False
                
            return True
        except Exception as e:
            logging.error(f"Context validation failed: {str(e)}")
            return False