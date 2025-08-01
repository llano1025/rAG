"""
Professional search result fusion using Reciprocal Rank Fusion (RRF) and other algorithms.
Implements industry-standard techniques for combining multiple search result sets.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict
import math

logger = logging.getLogger(__name__)

class ResultFusionConfig:
    """Configuration for result fusion algorithms."""
    
    def __init__(
        self,
        rrf_k: int = 60,
        text_weight: float = 0.6,
        semantic_weight: float = 0.4,
        score_normalization: bool = True,
        diversity_factor: float = 0.1
    ):
        self.rrf_k = rrf_k
        self.text_weight = text_weight
        self.semantic_weight = semantic_weight
        self.score_normalization = score_normalization
        self.diversity_factor = diversity_factor


def fuse_search_results(
    text_results: List[Dict[str, Any]],
    semantic_results: List[Dict[str, Any]],
    text_weight: float = 0.6,
    semantic_weight: float = 0.4,
    k_parameter: int = 60,
    max_results: int = 20,
    fusion_method: str = "rrf"
) -> List[Dict[str, Any]]:
    """
    Fuse text and semantic search results using professional algorithms.
    
    Args:
        text_results: Results from text-based search
        semantic_results: Results from semantic search
        text_weight: Weight for text search results
        semantic_weight: Weight for semantic search results
        k_parameter: RRF k parameter (typically 60)
        max_results: Maximum number of results to return
        fusion_method: Fusion algorithm ("rrf", "weighted", "comb_sum")
        
    Returns:
        List of fused search results sorted by combined score
    """
    try:
        if fusion_method == "rrf":
            return _reciprocal_rank_fusion(
                text_results, semantic_results, 
                text_weight, semantic_weight, k_parameter, max_results
            )
        elif fusion_method == "weighted":
            return _weighted_score_fusion(
                text_results, semantic_results,
                text_weight, semantic_weight, max_results
            )
        elif fusion_method == "comb_sum":
            return _combination_sum_fusion(
                text_results, semantic_results, max_results
            )
        else:
            logger.warning(f"Unknown fusion method: {fusion_method}, using RRF")
            return _reciprocal_rank_fusion(
                text_results, semantic_results,
                text_weight, semantic_weight, k_parameter, max_results
            )
            
    except Exception as e:
        logger.error(f"Result fusion failed: {str(e)}")
        # Fallback to simple combination
        return _simple_result_combination(text_results, semantic_results, max_results)


def _reciprocal_rank_fusion(
    text_results: List[Dict[str, Any]],
    semantic_results: List[Dict[str, Any]],
    text_weight: float,
    semantic_weight: float,
    k: int,
    max_results: int
) -> List[Dict[str, Any]]:
    """
    Implement Reciprocal Rank Fusion (RRF) algorithm.
    RRF is a state-of-the-art fusion method used in information retrieval.
    
    RRF Score = sum(weight / (k + rank)) for each ranking system
    """
    logger.info(f"Applying RRF fusion: text_weight={text_weight}, semantic_weight={semantic_weight}, k={k}")
    
    # Build document registry
    doc_registry = {}
    rrf_scores = defaultdict(float)
    
    # Process text search results
    for rank, result in enumerate(text_results, 1):
        doc_id = result.get('document_id')
        if doc_id:
            doc_registry[doc_id] = result
            rrf_score = text_weight / (k + rank)
            rrf_scores[doc_id] += rrf_score
    
    # Process semantic search results
    for rank, result in enumerate(semantic_results, 1):
        doc_id = result.get('document_id')
        if doc_id:
            if doc_id not in doc_registry:
                doc_registry[doc_id] = result
            rrf_score = semantic_weight / (k + rank)
            rrf_scores[doc_id] += rrf_score
    
    # Sort documents by RRF score
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Build final results
    fused_results = []
    for doc_id, rrf_score in sorted_docs[:max_results]:
        result = doc_registry[doc_id].copy()
        result['score'] = float(rrf_score)
        result['fusion_method'] = 'rrf'
        result['original_text_rank'] = _get_document_rank(doc_id, text_results)
        result['original_semantic_rank'] = _get_document_rank(doc_id, semantic_results)
        fused_results.append(result)
    
    logger.info(f"RRF fusion complete: {len(fused_results)} results from {len(text_results)} text + {len(semantic_results)} semantic")
    return fused_results


def _weighted_score_fusion(
    text_results: List[Dict[str, Any]],
    semantic_results: List[Dict[str, Any]],
    text_weight: float,
    semantic_weight: float,
    max_results: int
) -> List[Dict[str, Any]]:
    """
    Simple weighted score fusion.
    Combined Score = text_weight * text_score + semantic_weight * semantic_score
    """
    logger.info(f"Applying weighted score fusion: text_weight={text_weight}, semantic_weight={semantic_weight}")
    
    # Normalize scores to 0-1 range
    text_scores = _normalize_scores([r.get('score', 0.0) for r in text_results])
    semantic_scores = _normalize_scores([r.get('score', 0.0) for r in semantic_results])
    
    # Build document registry with normalized scores
    doc_registry = {}
    combined_scores = {}
    
    # Process text results
    for i, result in enumerate(text_results):
        doc_id = result.get('document_id')
        if doc_id:
            doc_registry[doc_id] = result
            combined_scores[doc_id] = text_weight * text_scores[i]
    
    # Process semantic results
    for i, result in enumerate(semantic_results):
        doc_id = result.get('document_id')
        if doc_id:
            if doc_id not in doc_registry:
                doc_registry[doc_id] = result
                combined_scores[doc_id] = 0.0
            combined_scores[doc_id] += semantic_weight * semantic_scores[i]
    
    # Sort by combined score
    sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Build final results
    fused_results = []
    for doc_id, combined_score in sorted_docs[:max_results]:
        result = doc_registry[doc_id].copy()
        result['score'] = float(combined_score)
        result['fusion_method'] = 'weighted'
        fused_results.append(result)
    
    return fused_results


def _combination_sum_fusion(
    text_results: List[Dict[str, Any]],
    semantic_results: List[Dict[str, Any]],
    max_results: int
) -> List[Dict[str, Any]]:
    """
    CombSum fusion: Simply sum the normalized scores.
    """
    logger.info("Applying CombSum fusion")
    
    # Build document registry
    doc_registry = {}
    combined_scores = defaultdict(float)
    
    # Normalize scores
    text_scores = _normalize_scores([r.get('score', 0.0) for r in text_results])
    semantic_scores = _normalize_scores([r.get('score', 0.0) for r in semantic_results])
    
    # Process text results
    for i, result in enumerate(text_results):
        doc_id = result.get('document_id')
        if doc_id:
            doc_registry[doc_id] = result
            combined_scores[doc_id] += text_scores[i]
    
    # Process semantic results
    for i, result in enumerate(semantic_results):
        doc_id = result.get('document_id')
        if doc_id:
            if doc_id not in doc_registry:
                doc_registry[doc_id] = result
            combined_scores[doc_id] += semantic_scores[i]
    
    # Sort by combined score
    sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Build final results
    fused_results = []
    for doc_id, combined_score in sorted_docs[:max_results]:
        result = doc_registry[doc_id].copy()
        result['score'] = float(combined_score)
        result['fusion_method'] = 'comb_sum'
        fused_results.append(result)
    
    return fused_results


def _normalize_scores(scores: List[float]) -> List[float]:
    """Normalize scores to 0-1 range using min-max normalization."""
    if not scores or len(scores) == 0:
        return []
    
    min_score = min(scores)
    max_score = max(scores)
    
    if max_score == min_score:
        return [1.0] * len(scores)  # All scores are the same
    
    return [(score - min_score) / (max_score - min_score) for score in scores]


def _get_document_rank(doc_id: str, results: List[Dict[str, Any]]) -> Optional[int]:
    """Get the rank (1-based) of a document in a result list."""
    for rank, result in enumerate(results, 1):
        if result.get('document_id') == doc_id:
            return rank
    return None


def _simple_result_combination(
    text_results: List[Dict[str, Any]],
    semantic_results: List[Dict[str, Any]],
    max_results: int
) -> List[Dict[str, Any]]:
    """
    Simple fallback combination: text results first, then semantic results.
    """
    logger.warning("Using simple result combination as fallback")
    
    seen_docs = set()
    combined_results = []
    
    # Add text results first
    for result in text_results:
        doc_id = result.get('document_id')
        if doc_id and doc_id not in seen_docs:
            result_copy = result.copy()
            result_copy['fusion_method'] = 'simple'
            combined_results.append(result_copy)
            seen_docs.add(doc_id)
            
            if len(combined_results) >= max_results:
                break
    
    # Add semantic results that weren't in text results
    for result in semantic_results:
        doc_id = result.get('document_id')
        if doc_id and doc_id not in seen_docs:
            result_copy = result.copy()
            result_copy['fusion_method'] = 'simple'
            combined_results.append(result_copy)
            seen_docs.add(doc_id)
            
            if len(combined_results) >= max_results:
                break
    
    return combined_results


def analyze_fusion_quality(
    text_results: List[Dict[str, Any]],
    semantic_results: List[Dict[str, Any]],
    fused_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Analyze the quality of fusion results.
    
    Returns:
        Dictionary with fusion quality metrics
    """
    try:
        # Calculate overlap between result sets
        text_docs = {r.get('document_id') for r in text_results}
        semantic_docs = {r.get('document_id') for r in semantic_results}
        fused_docs = {r.get('document_id') for r in fused_results}
        
        overlap = len(text_docs & semantic_docs)
        total_unique = len(text_docs | semantic_docs)
        
        # Calculate diversity metrics
        text_only = len(text_docs - semantic_docs)
        semantic_only = len(semantic_docs - text_docs)
        
        # Rank correlation metrics would go here in a full implementation
        
        return {
            'total_text_results': len(text_results),
            'total_semantic_results': len(semantic_results),
            'total_fused_results': len(fused_results),
            'overlap_count': overlap,
            'total_unique_documents': total_unique,
            'text_only_documents': text_only,
            'semantic_only_documents': semantic_only,
            'overlap_ratio': overlap / total_unique if total_unique > 0 else 0.0,
            'diversity_score': (text_only + semantic_only) / total_unique if total_unique > 0 else 0.0
        }
        
    except Exception as e:
        logger.error(f"Fusion quality analysis failed: {str(e)}")
        return {"error": str(e)}


# Professional fusion strategies for different query types
def get_fusion_strategy(query_type: str, query_confidence: float) -> Tuple[str, Dict[str, Any]]:
    """
    Recommend fusion strategy based on query characteristics.
    
    Args:
        query_type: Type of query (exact_phrase, multi_concept, etc.)
        query_confidence: Confidence in query processing (0.0-1.0)
        
    Returns:
        Tuple of (fusion_method, parameters)
    """
    if query_type == "exact_phrase":
        # For exact phrases, favor text search heavily
        return "rrf", {
            "text_weight": 0.8,
            "semantic_weight": 0.2,
            "k_parameter": 30
        }
    
    elif query_type == "multi_concept":
        # For complex conceptual queries, balance both approaches
        return "rrf", {
            "text_weight": 0.5,
            "semantic_weight": 0.5,
            "k_parameter": 60
        }
    
    elif query_type == "single_concept" and query_confidence > 0.7:
        # For clear single concepts, favor semantic search
        return "rrf", {
            "text_weight": 0.3,
            "semantic_weight": 0.7,
            "k_parameter": 45
        }
    
    elif query_type == "document_lookup":
        # For document/code lookups, heavily favor text search
        return "weighted", {
            "text_weight": 0.9,
            "semantic_weight": 0.1
        }
    
    else:
        # Default balanced approach
        return "rrf", {
            "text_weight": 0.6,
            "semantic_weight": 0.4,
            "k_parameter": 60
        }