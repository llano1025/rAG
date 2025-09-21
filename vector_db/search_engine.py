"""
Enhanced search engine with user-aware access control and hybrid search capabilities.
Uses Qdrant vector database and PostgreSQL for comprehensive document search.
"""

import logging
# asyncio removed - not used directly
import math
import re
import hashlib
import json
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime, timezone
import time
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from database.models import Document, DocumentChunk, User, SearchQuery, VectorIndex
from .storage_manager import VectorStorageManager, get_storage_manager, get_initialized_storage_manager
from .search_context_processor import ContextProcessor
from .reranker import get_reranker_manager, get_reranker_config, RerankResult
from .reranker.base_reranker import SearchResult as RerankerSearchResult
# QueryOptimizer removed - using direct Redis caching
from config import get_settings

# Lazy import for EmbeddingManager
def _get_embedding_manager():
    """Lazy import of EmbeddingManager."""
    try:
        from .embedding_manager import EmbeddingManager
        return EmbeddingManager
    except ImportError as e:
        logging.warning(f"EmbeddingManager not available: {e}")
        return None

logger = logging.getLogger(__name__)
settings = get_settings()

class SearchType:
    """Search type constants."""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    CONTEXTUAL = "contextual"

class CorpusStatsManager:
    """Manages corpus statistics for BM25 scoring with caching."""
    
    def __init__(self):
        self.stats_cache: Dict[str, Dict] = {}
        self.cache_expiry = 3600  # 1 hour in seconds
        self.stopwords: Set[str] = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
        }
    
    def preprocess_query(self, query: str) -> List[str]:
        """Preprocess query: lowercase, remove stopwords, basic tokenization."""
        # Convert to lowercase and tokenize
        tokens = re.findall(r'\b\w+\b', query.lower())
        
        # Remove stopwords and short tokens
        filtered_tokens = [token for token in tokens 
                          if token not in self.stopwords and len(token) > 2]
        
        return filtered_tokens if filtered_tokens else tokens[:3]  # Fallback to first 3 tokens
    
    async def get_corpus_stats(self, document_ids: List[int], db: Session) -> Dict[str, Any]:
        """Get cached or compute corpus statistics for BM25."""
        try:
            # Create cache key based on sorted document IDs
            cache_key = str(hash(tuple(sorted(document_ids))))
            
            if cache_key in self.stats_cache:
                stats = self.stats_cache[cache_key]
                # Check if cache is still valid (simple time-based for now)
                if len(self.stats_cache) < 100:  # Keep cache size reasonable
                    return stats
            
            logger.info(f"Computing corpus statistics for {len(document_ids)} documents")
            
            # Query all chunks for the documents
            chunks = db.query(DocumentChunk.text).filter(
                DocumentChunk.document_id.in_(document_ids)
            ).all()
            
            if not chunks:
                return self._get_default_stats()
            
            # Process all document texts
            doc_lengths = []
            term_doc_freq = defaultdict(int)  # How many documents contain each term
            total_docs = len(chunks)
            
            for chunk_row in chunks:
                text = chunk_row.text or ""
                tokens = self.preprocess_query(text)  # Reuse preprocessing logic
                doc_lengths.append(len(tokens))
                
                # Count unique terms in this document
                unique_terms = set(tokens)
                for term in unique_terms:
                    term_doc_freq[term] += 1
            
            # Calculate statistics
            avg_doc_length = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0
            
            # Calculate IDF for each term
            idf_values = {}
            for term, doc_freq in term_doc_freq.items():
                # BM25 IDF: log((N - df + 0.5) / (df + 0.5))
                idf = math.log((total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
                idf_values[term] = max(idf, 0.01)  # Prevent negative IDF
            
            stats = {
                'total_docs': total_docs,
                'avg_doc_length': avg_doc_length,
                'idf': idf_values,
                'computed_at': datetime.now().timestamp()
            }
            
            # Cache the results
            self.stats_cache[cache_key] = stats
            logger.info(f"Computed stats: {total_docs} docs, avg_len={avg_doc_length:.1f}, "
                       f"vocabulary_size={len(idf_values)}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to compute corpus statistics: {e}")
            return self._get_default_stats()
    
    def _get_default_stats(self) -> Dict[str, Any]:
        """Return default statistics when computation fails."""
        return {
            'total_docs': 1,
            'avg_doc_length': 100,
            'idf': defaultdict(lambda: 1.0),
            'computed_at': datetime.now().timestamp()
        }
    
    def calculate_bm25_score(self, query_terms: List[str], document_text: str, 
                           corpus_stats: Dict[str, Any], k1: float = 1.5, 
                           b: float = 0.75) -> float:
        """Calculate BM25 score for a document given query terms."""
        try:
            if not query_terms or not document_text.strip():
                return 0.0
            
            # Preprocess document
            doc_tokens = self.preprocess_query(document_text)
            doc_length = len(doc_tokens)
            avg_doc_length = corpus_stats.get('avg_doc_length', 100)
            idf_values = corpus_stats.get('idf', {})
            
            # Calculate term frequencies in document
            term_freq = Counter(doc_tokens)
            
            score = 0.0
            for term in query_terms:
                tf = term_freq.get(term, 0)
                if tf == 0:
                    continue
                
                # Get IDF for this term (default to small value for unseen terms)
                idf = idf_values.get(term, 0.5)
                
                # BM25 formula
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_length / avg_doc_length))
                
                term_score = idf * (numerator / denominator)
                score += term_score
            
            return max(score, 0.0)  # Ensure non-negative score
            
        except Exception as e:
            logger.error(f"BM25 calculation failed: {e}")
            # Fallback to simple term frequency scoring
            return sum(1 for term in query_terms if term in document_text.lower()) / len(query_terms)
    
    def normalize_scores(self, scores: List[float], method: str = "minmax") -> List[float]:
        """
        Normalize scores to [0,1] range.
        
        Args:
            scores: List of raw scores to normalize
            method: Normalization method ('minmax', 'sigmoid', or 'softmax')
            
        Returns:
            List of normalized scores in [0,1] range
        """
        if not scores:
            return []
        
        if len(scores) == 1:
            # Single score: use sigmoid normalization
            return [self._sigmoid_normalize(scores[0])]
        
        try:
            if method == "minmax":
                return self._minmax_normalize(scores)
            elif method == "sigmoid":
                return [self._sigmoid_normalize(score) for score in scores]
            elif method == "softmax":
                return self._softmax_normalize(scores)
            else:
                logger.warning(f"Unknown normalization method: {method}, using minmax")
                return self._minmax_normalize(scores)
                
        except Exception as e:
            logger.error(f"Score normalization failed: {e}")
            # Fallback: clip scores to [0,1] and warn
            normalized = [min(max(score, 0.0), 1.0) for score in scores]
            if any(score > 1.0 for score in scores):
                logger.warning("Some scores were clipped to 1.0 due to normalization failure")
            return normalized
    
    def _minmax_normalize(self, scores: List[float]) -> List[float]:
        """Min-max normalization: (x - min) / (max - min)."""
        if not scores:
            return []
            
        min_score = min(scores)
        max_score = max(scores)
        
        # Handle case where all scores are the same
        if max_score == min_score:
            # If all scores are 0, return 0; otherwise return 1 (perfect relevance)
            return [0.0 if max_score == 0.0 else 1.0] * len(scores)
        
        # Normalize to [0,1] range
        return [(score - min_score) / (max_score - min_score) for score in scores]
    
    def _sigmoid_normalize(self, score: float, scale: float = 2.0) -> float:
        """Sigmoid normalization: 1 / (1 + exp(-scale * score))."""
        try:
            return 1.0 / (1.0 + math.exp(-scale * score))
        except OverflowError:
            # Handle very large negative scores
            return 0.0 if score < 0 else 1.0
    
    def _softmax_normalize(self, scores: List[float]) -> List[float]:
        """Softmax normalization for probability distribution."""
        if not scores:
            return []
        
        # Prevent overflow by subtracting max score
        max_score = max(scores)
        exp_scores = []
        
        for score in scores:
            try:
                exp_scores.append(math.exp(score - max_score))
            except OverflowError:
                exp_scores.append(1e-10)  # Very small positive number
        
        total = sum(exp_scores)
        if total == 0:
            # Fallback to uniform distribution
            return [1.0 / len(scores)] * len(scores)
        
        return [exp_score / total for exp_score in exp_scores]

class TagMatchMode:
    """Tag matching modes for search filtering."""
    ANY = "any"  # Document must have ANY of the specified tags (OR logic)
    ALL = "all"  # Document must have ALL of the specified tags (AND logic)
    EXACT = "exact"  # Document must have EXACTLY the specified tags (no more, no less)

class SearchFilter:
    """Search filter class for structured filtering."""
    
    def __init__(self):
        self.user_id: Optional[int] = None
        self.document_ids: Optional[List[int]] = None
        self.content_types: Optional[List[str]] = None
        self.date_range: Optional[Tuple[datetime, datetime]] = None
        self.file_size_range: Optional[Tuple[int, int]] = None
        self.tags: Optional[List[str]] = None
        self.tag_match_mode: str = TagMatchMode.ANY
        self.exclude_tags: Optional[List[str]] = None  # Tags to exclude (NOT logic)
        self.language: Optional[str] = None
        self.is_public: Optional[bool] = None
        self.min_score: Optional[float] = None
        
        # Reranker settings
        self.enable_reranking: bool = True
        self.reranker_model: Optional[str] = None
        self.rerank_score_weight: float = 0.5
        self.min_rerank_score: Optional[float] = None
        self.max_results_to_rerank: int = 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert filter to dictionary for serialization."""
        return {
            'user_id': self.user_id,
            'document_ids': self.document_ids,
            'content_types': self.content_types,
            'date_range': [dt.isoformat() for dt in self.date_range] if self.date_range else None,
            'file_size_range': self.file_size_range,
            'tags': self.tags,
            'tag_match_mode': self.tag_match_mode,
            'exclude_tags': self.exclude_tags,
            'language': self.language,
            'is_public': self.is_public,
            'min_score': self.min_score,
            'enable_reranking': self.enable_reranking,
            'reranker_model': self.reranker_model,
            'rerank_score_weight': self.rerank_score_weight,
            'min_rerank_score': self.min_rerank_score,
            'max_results_to_rerank': self.max_results_to_rerank
        }
    
    def normalize_tags(self, tags: List[str]) -> List[str]:
        """Normalize tags for consistent matching (lowercase, strip whitespace)."""
        if not tags:
            return []
        
        normalized = []
        for tag in tags:
            if isinstance(tag, str) and tag.strip():
                normalized_tag = tag.strip().lower()
                if normalized_tag not in normalized:  # Remove duplicates
                    normalized.append(normalized_tag)
        
        return normalized
    
    def set_tags(self, tags: Optional[List[str]]):
        """Set tags with automatic normalization."""
        if tags:
            self.tags = self.normalize_tags(tags)
        else:
            self.tags = None
    
    def set_exclude_tags(self, exclude_tags: Optional[List[str]]):
        """Set exclude tags with automatic normalization.""" 
        if exclude_tags:
            self.exclude_tags = self.normalize_tags(exclude_tags)
        else:
            self.exclude_tags = None

class SearchResult:
    """Enhanced search result with metadata and access control."""
    
    def __init__(
        self,
        chunk_id: str,
        document_id: int,
        text: str,
        score: float,
        metadata: Dict[str, Any],
        document_metadata: Dict[str, Any] = None,
        highlight: str = None
    ):
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.text = text
        self.score = score
        self.metadata = metadata
        self.document_metadata = document_metadata or {}
        self.highlight = highlight
        self.timestamp = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'chunk_id': self.chunk_id,
            'document_id': self.document_id,
            'text': self.text,
            'score': self.score,
            'metadata': self.metadata,
            'document_metadata': self.document_metadata,
            'highlight': self.highlight,
            'timestamp': self.timestamp.isoformat()
        }

class EnhancedSearchEngine:
    """
    Enhanced search engine with user-aware access control and hybrid search.
    
    Features:
    - User-aware document access control
    - Hybrid semantic + keyword search
    - Contextual search with content + context vectors
    - Search result caching
    - Search analytics and logging
    - Multi-index search across user documents
    """
    
    def __init__(self, storage_manager: VectorStorageManager = None, embedding_manager = None):
        """Initialize enhanced search engine."""
        self.storage_manager = storage_manager or get_storage_manager()
        self.embedding_manager = embedding_manager  # Use provided or initialize lazily
        self.context_processor = ContextProcessor()
        
        # Ensure storage manager is initialized (will be called async)
        self._storage_initialized = False
        
        # Reranker components
        self.reranker_manager = get_reranker_manager()
        self.reranker_config = get_reranker_config()
        
        # BM25 corpus statistics manager
        self.corpus_stats_manager = CorpusStatsManager()
        
        # Redis cache manager for caching and performance
        from utils.caching.redis_manager import RedisManager
        self.redis_manager = RedisManager()
        
        # Search configuration
        self.default_limit = 10
        self.max_limit = 100
        self.cache_duration_hours = 24
    
    async def _ensure_storage_initialized(self):
        """Ensure storage manager is properly initialized."""
        if not self._storage_initialized:
            logger.info("Initializing storage manager for search operations")
            success = await self.storage_manager.initialize()
            if success:
                self._storage_initialized = True
                logger.info("Storage manager initialized successfully")
            else:
                logger.error("Failed to initialize storage manager")
                raise RuntimeError("Storage manager initialization failed")
    
    def _get_embedding_manager_instance(self):
        """Get embedding manager instance, initializing if needed."""
        if self.embedding_manager is None:
            EmbeddingManager = _get_embedding_manager()
            if EmbeddingManager:
                self.embedding_manager = EmbeddingManager()
            else:
                raise ImportError("EmbeddingManager not available")
        return self.embedding_manager
    
    async def search_with_context(
        self,
        query: str,
        search_type: str = SearchType.SEMANTIC,
        user_id: int = None,
        top_k: int = None,
        db: Session = None,
        use_cache: bool = True,
        # Reranker parameters for chat controller compatibility
        enable_reranking: bool = True,
        reranker_model: Optional[str] = None,
        rerank_score_weight: float = 0.5,
        min_rerank_score: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform search with context - compatibility method for chat controller.
        
        Args:
            query: Search query text
            search_type: Type of search (semantic, keyword, hybrid, contextual)
            user_id: ID of user performing the search
            top_k: Maximum number of results (aliases for limit)
            db: Database session
            use_cache: Whether to use cached results
            
        Returns:
            List of search results as dictionaries
        """
        try:
            # Get user from database
            if not db or not user_id:
                logger.error("Database session and user_id required for search_with_context")
                return []
            
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                logger.error(f"User {user_id} not found")
                return []
            
            # Convert top_k to limit
            limit = top_k or self.default_limit
            
            # Create search filter with reranker settings
            search_filters = SearchFilter()
            search_filters.enable_reranking = enable_reranking
            search_filters.reranker_model = reranker_model
            search_filters.rerank_score_weight = rerank_score_weight
            search_filters.min_rerank_score = min_rerank_score
            
            # Perform search using existing method
            results = await self.search(
                query=query,
                user=user,
                search_type=search_type,
                filters=search_filters,
                limit=limit,
                db=db,
                use_cache=use_cache
            )
            
            # Convert SearchResult objects to dictionaries for compatibility
            dict_results = []
            for result in results:
                dict_result = {
                    "document_id": result.document_id,
                    "chunk_id": result.chunk_id,
                    "text": result.text,
                    "similarity_score": result.score,
                    "metadata": result.metadata
                }
                dict_results.append(dict_result)
            
            logger.info(f"search_with_context completed: {len(dict_results)} results for query '{query[:50]}...'")
            return dict_results
            
        except Exception as e:
            logger.error(f"search_with_context failed: {str(e)}")
            return []

    def _generate_stable_cache_key(self, query: str, user_id: int, search_type: str, filters: 'SearchFilter') -> str:
        """Generate a stable, deterministic cache key for search operations."""
        try:
            # Create normalized filter dictionary with only essential stable fields
            normalized_filter = {
                'search_type': search_type,  # Include search type for cache separation
                'user_id': filters.user_id,
                'document_ids': sorted(filters.document_ids) if filters.document_ids else None,
                'content_types': sorted(filters.content_types) if filters.content_types else None,
                'date_range': [dt.isoformat() for dt in filters.date_range] if filters.date_range else None,
                'file_size_range': filters.file_size_range,
                'tags': sorted(filters.tags) if filters.tags else None,
                'language': filters.language,
                'is_public': filters.is_public,
                'min_score': filters.min_score,
                # Reranker settings that affect results
                'enable_reranking': filters.enable_reranking,
                'reranker_model': filters.reranker_model,
                'rerank_score_weight': filters.rerank_score_weight,
                'min_rerank_score': filters.min_rerank_score,
            }
            
            # Remove None values and normalize the structure
            stable_filter = {k: v for k, v in normalized_filter.items() if v is not None}
            
            # Sort keys for consistent ordering
            filter_str = json.dumps(stable_filter, sort_keys=True, ensure_ascii=True)
            
            # Create stable hash
            hash_input = f"{query.strip()}_{user_id}_{search_type}_{filter_str}"
            cache_hash = hashlib.md5(hash_input.encode('utf-8')).hexdigest()
            cache_key = f"search_{search_type}_{cache_hash}"
            
            # Debug logging to identify instability sources  
            logger.info(f"Cache key generation:")
            logger.info(f"Query: '{query[:50]}...'")
            logger.info(f"User ID: {user_id}")
            logger.info(f"Search Type: {search_type}")
            logger.info(f"Normalized filter: {stable_filter}")
            logger.info(f"Filter string: {filter_str}")
            logger.info(f"Hash input length: {len(hash_input)} chars")
            logger.info(f"Final key: {cache_key}")
            
            return cache_key
        except Exception as e:
            logger.warning(f"Cache key generation failed, using fallback: {e}")
            # Fallback to simple key without filter hash
            fallback_key = f"search_simple_{hashlib.md5(f'{query.strip()}_{user_id}'.encode()).hexdigest()}"
            logger.debug(f"   Fallback key: {fallback_key}")
            return fallback_key

    async def search(
        self,
        query: str,
        user: User,
        search_type: str = SearchType.SEMANTIC,
        filters: SearchFilter = None,
        limit: int = None,
        db: Session = None,
        use_cache: bool = True
    ) -> List[SearchResult]:
        """
        Perform comprehensive search with user access control.
        
        Args:
            query: Search query text
            user: User performing the search
            search_type: Type of search (semantic, keyword, hybrid, contextual)
            filters: Additional search filters
            limit: Maximum number of results
            db: Database session
            use_cache: Whether to use cached results
            
        Returns:
            List of search results
        """
        start_time = datetime.now(timezone.utc)
        limit = min(limit or self.default_limit, self.max_limit)
        filters = filters or SearchFilter()
        
        # Only initialize storage for vector-based search types
        vector_search_types = {SearchType.SEMANTIC, SearchType.HYBRID, SearchType.CONTEXTUAL}
        database_only_search_types = {SearchType.KEYWORD, "basic"}  # Include "basic" alias for keyword
        
        if search_type in vector_search_types:
            await self._ensure_storage_initialized()
        elif search_type in database_only_search_types:
            logger.info(f"Skipping Qdrant initialization for {search_type} search (database-only)")
        else:
            # Unknown search type - be safe and initialize storage
            logger.warning(f"Unknown search type '{search_type}', initializing storage as safety measure")
            await self._ensure_storage_initialized()
        
        # Generate cache key ONCE at the beginning - before any filter modifications
        cache_key = None
        if use_cache:
            cache_key = self._generate_stable_cache_key(query, user.id, search_type, filters)
            logger.info(f"Generated cache key for entire method: {cache_key}")
        
        try:
            # Hierarchical cache strategy - check multiple cache levels
            if use_cache and cache_key:
                # Check end-to-end Redis cache first (fastest)
                try:
                    logger.info(f"Checking Redis cache for key: {cache_key}")
                    logger.info(f"Query: '{query[:50]}...', User: {user.id}, Filters: {len(filters.to_dict())} properties")
                    cached_data = await self.redis_manager.get_value(cache_key)
                    if cached_data:
                        logger.info(f"CACHE HIT! Found {len(cached_data)} cached results for query: {query[:50]}...")
                        # Convert back to SearchResult objects
                        cached_results = []
                        for result_data in cached_data:
                            result = SearchResult(
                                chunk_id=result_data['chunk_id'],
                                document_id=result_data['document_id'],
                                text=result_data['text'],
                                score=result_data['score'],
                                metadata=result_data['metadata'],
                                document_metadata=result_data.get('document_metadata', {}),
                                highlight=result_data.get('highlight')
                            )
                            cached_results.append(result)
                        logger.info(f"Returning {len(cached_results)} Redis cached results")
                        return cached_results
                    else:
                        logger.info(f"CACHE MISS - No Redis cached results found")
                except Exception as e:
                    logger.warning(f"Redis cache retrieval failed: {e}")
                
                # Fallback to database cache
                cached_results = await self._get_cached_results(query, user.id, search_type, filters, db)
                if cached_results:
                    logger.info(f"Returning database cached results for query: {query[:50]}...")
                    return cached_results
                
                # Check vector cache coverage for semantic searches
                # This optimizes cases where vector operations are cached but final results aren't
                if search_type == SearchType.SEMANTIC:
                    # Get accessible documents first for coverage check
                    filters.user_id = user.id
                    accessible_doc_ids = await self._get_accessible_documents(user, db, filters)
                    filters.document_ids = accessible_doc_ids
                    
                    # Check if we have high vector cache coverage
                    vector_coverage = await self._check_vector_cache_coverage(query, filters, accessible_doc_ids)
                    
                    if vector_coverage >= 0.8:  # 80% or more documents have cached vector results
                        assembly_start = time.time()
                        logger.info(f"High vector cache coverage ({vector_coverage:.1%}), assembling from cache")
                        assembled_results = await self._assemble_from_vector_cache(query, filters, limit, accessible_doc_ids)
                        
                        if assembled_results:
                            assembly_time = (time.time() - assembly_start) * 1000
                            
                            # Cache these assembled results for future end-to-end cache hits
                            try:
                                logger.info(f"Caching {len(assembled_results)} assembled results for key: {cache_key}")
                                serialized_results = [result.to_dict() for result in assembled_results]
                                success = await self.redis_manager.set_value(
                                    cache_key,
                                    serialized_results,
                                    ttl=self.cache_duration_hours * 3600
                                )
                                if success:
                                    logger.info(f"Successfully cached assembled results")
                                else:
                                    logger.warning(f"Failed to cache assembled results")
                                logger.info(f"Vector cache optimization: assembled {len(assembled_results)} results in {assembly_time:.1f}ms (saved full search)")
                            except Exception as e:
                                logger.warning(f"Failed to cache assembled results: {e}")
                            
                            return assembled_results
                        else:
                            logger.info(f"Vector cache assembly returned no results, falling back to full search")
            
            # Apply user access control to filters (skip if already done for semantic cache check)
            if not hasattr(filters, 'document_ids') or not filters.document_ids:
                filters.user_id = user.id
                accessible_doc_ids = await self._get_accessible_documents(user, db, filters)
                filters.document_ids = accessible_doc_ids
            else:
                accessible_doc_ids = filters.document_ids
            
            if not accessible_doc_ids:
                logger.info(f"No accessible documents for user {user.id}")
                return []
            
            # Perform search based on type
            if search_type == SearchType.SEMANTIC:
                results = await self._semantic_search(query, filters, limit, db)
            elif search_type == SearchType.KEYWORD or search_type == "basic":
                results = await self._keyword_search(query, filters, limit, db)
            elif search_type == SearchType.HYBRID:
                results = await self._hybrid_search(query, filters, limit, db)
            elif search_type == SearchType.CONTEXTUAL:
                results = await self._contextual_search(query, filters, limit, db)
            else:
                raise ValueError(f"Unknown search type: {search_type}")
            
            # Apply reranking if enabled and configured
            if filters.enable_reranking and self.reranker_config.enabled:
                results = await self._apply_reranking(query, results, filters, search_type)
            
            # Log search query and cache results
            search_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            await self._log_search_query(query, user.id, search_type, filters, len(results), search_time, db)
            
            if use_cache and cache_key and results:
                # Cache results using both methods for optimal performance
                # Database cache for analytics and persistence
                await self._cache_results(query, user.id, search_type, filters, results, db)
                
                # Redis cache via query optimizer for fast retrieval
                try:
                    logger.info(f"Caching {len(results)} search results for key: {cache_key}")
                    logger.info(f"Confirming same cache key used for storage: {cache_key}")
                    serialized_results = [result.to_dict() for result in results]
                    success = await self.redis_manager.set_value(
                        cache_key,
                        serialized_results,
                        ttl=self.cache_duration_hours * 3600
                    )
                    if success:
                        logger.info(f"Successfully cached {len(results)} search results")
                    else:
                        logger.warning(f"Failed to cache search results")
                except Exception as e:
                    logger.warning(f"Failed to cache results in Redis: {e}")
            
            logger.info(f"Search completed in {search_time:.2f}ms, returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            return []
    
    async def _get_accessible_documents(self, user: User, db: Session, filters: SearchFilter = None) -> List[int]:
        """Get list of document IDs accessible to the user with optional filtering."""
        try:
            # Build query for accessible documents with active vector indices
            query = db.query(Document.id).join(
                VectorIndex, Document.id == VectorIndex.document_id
            ).filter(
                Document.is_deleted == False,
                VectorIndex.is_active == True,  # Only documents with active vector indices
                or_(
                    Document.user_id == user.id,  # User's own documents
                    Document.is_public == True,   # Public documents
                    and_(
                        user.is_superuser == True  # Admin access
                    )
                )
            ).distinct()  # Remove duplicates from join
            
            # Additional filter for admin users
            if user.has_role("admin"):
                # Admins can access all documents with active indices
                query = db.query(Document.id).join(
                    VectorIndex, Document.id == VectorIndex.document_id
                ).filter(
                    Document.is_deleted == False,
                    VectorIndex.is_active == True
                ).distinct()
            
            # Apply additional filters if provided
            if filters:
                query = self._apply_search_filters_to_query(query, filters)
            
            doc_ids = [row[0] for row in query.all()]
            logger.debug(f"User {user.id} has access to {len(doc_ids)} documents (after filtering)")
            
            return doc_ids
            
        except Exception as e:
            logger.error(f"Failed to get accessible documents: {e}")
            return []
    
    def _apply_search_filters_to_query(self, query, filters: SearchFilter):
        """Apply search filters to a database query."""
        
        # Apply content type filtering
        if filters.content_types:
            query = query.filter(Document.content_type.in_(filters.content_types))
        
        # Apply date range filtering
        if filters.date_range and len(filters.date_range) == 2:
            start_date, end_date = filters.date_range
            query = query.filter(
                Document.created_at >= start_date,
                Document.created_at <= end_date
            )
        
        # Apply file size filtering
        if filters.file_size_range and len(filters.file_size_range) == 2:
            min_size, max_size = filters.file_size_range
            query = query.filter(
                Document.file_size >= min_size,
                Document.file_size <= max_size
            )
        
        # Apply language filtering
        if filters.language:
            query = query.filter(Document.language == filters.language)
        
        # Apply public/private filtering
        if filters.is_public is not None:
            query = query.filter(Document.is_public == filters.is_public)
        
        # Apply tag filtering - this is the main enhancement
        if filters.tags or filters.exclude_tags:
            query = self._apply_tag_filters_to_query(query, filters)
        
        return query
    
    def _apply_tag_filters_to_query(self, query, filters: SearchFilter):
        """Apply tag filtering to a database query optimized for TEXT columns."""
        from sqlalchemy import text
        import logging
        
        logger = logging.getLogger(__name__)
        
        try:
            # Apply include tags filtering
            if filters.tags:
                normalized_tags = filters.normalize_tags(filters.tags)
                
                if filters.tag_match_mode == TagMatchMode.ANY:
                    # Document must have ANY of the specified tags (OR logic)
                    # Use LIKE patterns for TEXT column with JSON strings
                    conditions = []
                    for tag in normalized_tags:
                        conditions.append(f"LOWER(documents.tags) LIKE LOWER('%\"{tag}\"%')")
                    
                    if conditions:
                        query = query.filter(text(f"({' OR '.join(conditions)})"))
                
                elif filters.tag_match_mode == TagMatchMode.ALL:
                    # Document must have ALL of the specified tags (AND logic)
                    conditions = []
                    for tag in normalized_tags:
                        conditions.append(f"LOWER(documents.tags) LIKE LOWER('%\"{tag}\"%')")
                    
                    if conditions:
                        query = query.filter(text(f"({' AND '.join(conditions)})"))
                
                elif filters.tag_match_mode == TagMatchMode.EXACT:
                    # Document must have EXACTLY the specified tags (no more, no less)
                    # Generate different possible JSON formats
                    sorted_tags = sorted(normalized_tags)
                    
                    # Format variations for JSON arrays
                    formats = [
                        '["' + '","'.join(sorted_tags) + '"]',  # Standard format with spaces
                        "[" + ",".join([f'"{tag}"' for tag in sorted_tags]) + "]",  # Compact format
                        '[ "' + '", "'.join(sorted_tags) + '" ]',  # Spaced format
                    ]
                    
                    conditions = []
                    for i, _ in enumerate(formats):
                        conditions.append(f"documents.tags = :format_{i}")
                    
                    # Also check normalized version (remove spaces and quotes)
                    normalized_check = "[" + ",".join(sorted_tags) + "]"
                    conditions.append("REPLACE(REPLACE(documents.tags, ' ', ''), '\"', '') = :normalized")
                    
                    params = {f"format_{i}": fmt for i, fmt in enumerate(formats)}
                    params["normalized"] = normalized_check
                    
                    query = query.filter(text(f"({' OR '.join(conditions)})")).params(**params)
            
            # Apply exclude tags filtering (NOT logic)
            if filters.exclude_tags:
                normalized_exclude_tags = filters.normalize_tags(filters.exclude_tags)
                
                # Document must NOT contain any of the excluded tags
                conditions = []
                for tag in normalized_exclude_tags:
                    conditions.append(f"LOWER(documents.tags) NOT LIKE LOWER('%\"{tag}\"%')")
                
                if conditions:
                    query = query.filter(text(f"({' AND '.join(conditions)})"))
            
            return query
            
        except Exception as e:
            logger.error(f"Error applying tag filters: {e}")
            # Return original query if tag filtering fails
            return query
    
    async def _semantic_search(
        self,
        query: str,
        filters: SearchFilter,
        limit: int,
        db: Session
    ) -> List[SearchResult]:
        """Perform intelligent multi-embedding semantic search."""
        try:
            # Import query processor for intelligent search
            from utils.search.query_processor import get_query_processor
            
            embedding_manager = self._get_embedding_manager_instance()
            query_processor = get_query_processor()
            
            # Process query to extract semantic components
            processed_query = query_processor.process_query(query)
            
            logger.info(f"Semantic search with strategy: {processed_query.search_strategy}, "
                       f"concepts: {processed_query.primary_terms}, metadata: {processed_query.secondary_terms}")
            
            # Generate multiple embeddings based on query strategy
            embeddings_to_search = []
            
            # Always include full query embedding
            full_query_embeddings = await embedding_manager.generate_embeddings([query])
            embeddings_to_search.append(("full_query", full_query_embeddings[0], 0.4))
            
            # Add concept-based embeddings if we have primary terms
            if processed_query.primary_terms:
                concepts_text = " ".join(processed_query.primary_terms)
                concept_embeddings = await embedding_manager.generate_embeddings([concepts_text])
                embeddings_to_search.append(("concepts", concept_embeddings[0], 0.6))
            
            # Add expanded query embedding for better recall
            if processed_query.confidence < 0.8:
                expanded_query = query_processor.expand_query(processed_query)
                expanded_text = " ".join(expanded_query.primary_terms)
                if expanded_text != query and len(expanded_text.strip()) > 0:
                    expanded_embeddings = await embedding_manager.generate_embeddings([expanded_text])
                    embeddings_to_search.append(("expanded", expanded_embeddings[0], 0.3))
            
            # Search with multiple embeddings
            all_results = {}  # unique_key -> best result
            
            for doc_id in filters.document_ids:
                index_name = f"doc_{doc_id}"
                doc_results = {}  # chunk_id -> best score
                
                # Search with each embedding strategy
                for strategy_name, query_vector, weight in embeddings_to_search:
                    try:
                        # Direct vector search via storage manager
                        search_results = await self.storage_manager.search_vectors(
                            index_name=index_name,
                            query_vector=query_vector,
                            vector_type='content',
                            limit=limit * 2,  # Get more results for fusion
                            score_threshold=filters.min_score or 0.1,
                            metadata_filters={'user_id': filters.user_id} if filters.user_id else None
                        )
                        logger.info(f"Initiating search using Qdrant...")
                        results = search_results

                        # Process results with weighted scoring
                        for result in results:
                            chunk_id = result.get('metadata', {}).get('chunk_id')
                            if chunk_id:
                                weighted_score = result.get('score', 0.0) * weight
                                
                                if chunk_id not in doc_results or weighted_score > doc_results[chunk_id]['score']:
                                    doc_results[chunk_id] = {
                                        'result': result,
                                        'score': weighted_score,
                                        'strategy': strategy_name
                                    }
                                    
                    except Exception as e:
                        logger.warning(f"Search failed for {strategy_name} on doc {doc_id}: {e}")
                        continue
                
                # Convert best results for this document
                for chunk_data in doc_results.values():
                    search_result = await self._create_search_result(chunk_data['result'], db)
                    if search_result:
                        # Update score with weighted value
                        search_result.score = chunk_data['score']
                        search_result.metadata['search_strategy'] = chunk_data['strategy']
                        
                        result_key = f"{search_result.document_id}_{search_result.chunk_id}"
                        if result_key not in all_results or search_result.score > all_results[result_key].score:
                            all_results[result_key] = search_result
            
            # Convert to list and sort by score
            final_results = list(all_results.values())
            final_results.sort(key=lambda x: x.score, reverse=True)
            
            logger.info(f"Multi-embedding semantic search completed: {len(final_results)} results")
            return final_results[:limit]
            
        except Exception as e:
            logger.error(f"Multi-embedding semantic search failed: {e}")
            # Fallback to simple semantic search
            return await self._simple_semantic_search(query, filters, limit, db)
    
    async def _simple_semantic_search(
        self,
        query: str,
        filters: SearchFilter,
        limit: int,
        db: Session
    ) -> List[SearchResult]:
        """Simple fallback semantic search."""
        try:
            # Generate query embedding
            embedding_manager = self._get_embedding_manager_instance()
            query_embeddings = await embedding_manager.generate_embeddings([query])
            query_vector = query_embeddings[0]
            
            # Search across all accessible document indices
            all_results = []
            
            for doc_id in filters.document_ids:
                index_name = f"doc_{doc_id}"
                
                # Search content vectors with Qdrant
                results = await self.storage_manager.search_vectors(
                    index_name=index_name,
                    query_vector=query_vector,
                    vector_type="content",
                    limit=limit,
                    score_threshold=filters.min_score
                )
                
                # Convert to SearchResult objects
                for result in results:
                    search_result = await self._create_search_result(result, db)
                    if search_result:
                        all_results.append(search_result)
            
            # Sort by score and return top results
            all_results.sort(key=lambda x: x.score, reverse=True)
            return all_results[:limit]
            
        except Exception as e:
            logger.error(f"Simple semantic search failed: {e}")
            return []
    
    async def _contextual_search(
        self,
        query: str,
        filters: SearchFilter,
        limit: int,
        db: Session
    ) -> List[SearchResult]:
        """Perform contextual search combining content and context vectors."""
        try:
            # Generate query embedding
            embedding_manager = self._get_embedding_manager_instance()
            query_embeddings = await embedding_manager.generate_embeddings([query])
            query_vector = query_embeddings[0]
            
            # Search across all accessible document indices
            all_results = []
            
            for doc_id in filters.document_ids:
                index_name = f"doc_{doc_id}"
                
                # Perform contextual search
                results = await self.storage_manager.contextual_search(
                    index_name=index_name,
                    query_vector=query_vector,
                    limit=limit,
                    content_weight=0.7,
                    context_weight=0.3,
                    score_threshold=filters.min_score
                )
                
                # Convert to SearchResult objects
                for result in results:
                    search_result = await self._create_search_result_from_contextual(result, db)
                    if search_result:
                        all_results.append(search_result)
            
            # Sort by combined score and return top results
            all_results.sort(key=lambda x: x.score, reverse=True)
            return all_results[:limit]
            
        except Exception as e:
            logger.error(f"Contextual search failed: {e}")
            return []
    
    async def _keyword_search(
        self,
        query: str,
        filters: SearchFilter,
        limit: int,
        db: Session
    ) -> List[SearchResult]:
        """
        Enhanced keyword search using hybrid database filtering + BM25 scoring.
        
        Two-stage approach:
        1. Fast database prefiltering with ILIKE queries
        2. BM25 reranking for statistical relevance
        """
        try:
            logger.info(f"Enhanced keyword search: '{query}' (limit={limit})")
            
            # Preprocess query and fast database filtering
            query_terms = self.corpus_stats_manager.preprocess_query(query)
            if not query_terms:
                logger.warning(f"No valid terms found in query: '{query}'")
                return []
            
            logger.debug(f"Processed query terms: {query_terms}")
            
            # Build optimized database query (get more candidates for reranking)
            candidate_limit = min(limit * 3, 100)  # 3x for reranking, max 100
            base_query = db.query(DocumentChunk, Document).join(
                Document, DocumentChunk.document_id == Document.id
            ).filter(
                Document.id.in_(filters.document_ids),
                Document.is_deleted == False
            )
            
            # Create text search filters for each term
            text_filters = []
            for term in query_terms:
                text_filters.append(DocumentChunk.text.ilike(f"%{term}%"))
            
            if text_filters:
                base_query = base_query.filter(or_(*text_filters))
            
            # Apply minimum score filter if specified
            # Note: minimum score filtering is applied after BM25 scoring and normalization
            # (see lines ~1024 where normalized scores are filtered)
            
            # Execute fast prefiltering query
            candidates = base_query.limit(candidate_limit).all()
            logger.debug(f"Database prefiltering returned {len(candidates)} candidates")
            
            if not candidates:
                return []
            
            # Get corpus statistics for BM25
            corpus_stats = await self.corpus_stats_manager.get_corpus_stats(
                filters.document_ids, db
            )
            
            # BM25 scoring and reranking
            scored_results = []
            raw_scores = []
            
            for chunk, document in candidates:
                # Calculate BM25 score
                bm25_score = self.corpus_stats_manager.calculate_bm25_score(
                    query_terms, chunk.text or "", corpus_stats
                )
                
                # Store raw scores for later normalization
                raw_scores.append(bm25_score)
                
                # Create SearchResult with raw BM25 score (will be normalized later)
                search_result = SearchResult(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    text=chunk.text,
                    score=bm25_score,  # Temporary raw score
                    metadata={
                        'chunk_index': chunk.chunk_index,
                        'text_length': chunk.text_length,
                        'start_char': chunk.start_char,
                        'end_char': chunk.end_char,
                        'search_algorithm': 'bm25',
                        'query_terms': query_terms,
                        'raw_bm25_score': bm25_score  # Store raw score for debugging
                    },
                    document_metadata={
                        'filename': document.filename,
                        'content_type': document.content_type,
                        'created_at': document.created_at.isoformat(),
                        'version': document.version,
                        'tags': document.get_tag_list()  # Include document tags
                    },
                    highlight=self._generate_highlight(query, chunk.text)
                )
                scored_results.append(search_result)
            
            # Normalize scores to [0,1] range
            if scored_results and raw_scores:
                normalized_scores = self.corpus_stats_manager.normalize_scores(
                    raw_scores, method="minmax"
                )
                
                # Update scores with normalized values and apply minimum score filter
                filtered_results = []
                for result, normalized_score in zip(scored_results, normalized_scores):
                    result.score = normalized_score
                    
                    # Apply minimum score filter on normalized scores
                    if filters.min_score is None or normalized_score >= filters.min_score:
                        filtered_results.append(result)
                
                scored_results = filtered_results
            
            # Sort by normalized score and return top results
            scored_results.sort(key=lambda x: x.score, reverse=True)
            final_results = scored_results[:limit]
            
            logger.info(f"BM25 keyword search completed: {len(final_results)} results "
                       f"(avg_score={sum(r.score for r in final_results)/len(final_results):.3f} "
                       f"if final_results else 0)")
            
            return final_results
            
        except Exception as e:
            logger.error(f"BM25 keyword search failed: {e}")
            return []
    
    async def _hybrid_search(
        self,
        query: str,
        filters: SearchFilter,
        limit: int,
        db: Session
    ) -> List[SearchResult]:
        """Perform hybrid search combining semantic and keyword approaches."""
        try:
            # Perform both semantic and keyword searches
            semantic_results = await self._semantic_search(query, filters, limit, db)
            keyword_results = await self._keyword_search(query, filters, limit, db)
            
            # Combine and re-rank results
            combined_results = {}
            
            # Add semantic results with weight
            for result in semantic_results:
                key = f"{result.document_id}_{result.chunk_id}"
                combined_results[key] = {
                    'result': result,
                    'semantic_score': result.score,
                    'keyword_score': 0.0
                }
            
            # Add keyword results with weight
            for result in keyword_results:
                key = f"{result.document_id}_{result.chunk_id}"
                if key in combined_results:
                    combined_results[key]['keyword_score'] = result.score
                else:
                    combined_results[key] = {
                        'result': result,
                        'semantic_score': 0.0,
                        'keyword_score': result.score
                    }
            
            # Calculate hybrid scores and apply min_score filtering
            final_results = []
            semantic_weight = 0.7
            keyword_weight = 0.3
            min_score_threshold = filters.min_score or 0.0
            
            for key, data in combined_results.items():
                hybrid_score = (
                    data['semantic_score'] * semantic_weight +
                    data['keyword_score'] * keyword_weight
                )
                
                # Apply min_score filtering on the final hybrid score
                if hybrid_score < min_score_threshold:
                    logger.debug(f"Filtered hybrid result {key} (score: {hybrid_score:.3f} < min_score: {min_score_threshold})")
                    continue
                
                result = data['result']
                result.score = hybrid_score
                result.metadata['semantic_score'] = data['semantic_score']
                result.metadata['keyword_score'] = data['keyword_score']
                result.metadata['hybrid_score'] = hybrid_score
                
                final_results.append(result)
            
            # Sort by hybrid score and return top results
            final_results.sort(key=lambda x: x.score, reverse=True)
            return final_results[:limit]
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []
    
    async def _create_search_result(self, vector_result: Dict, db: Session) -> Optional[SearchResult]:
        """Create SearchResult from vector search result."""
        try:
            metadata = vector_result.get('metadata', {})
            chunk_id = metadata.get('chunk_id')
            
            if not chunk_id:
                return None
            
            # Get chunk from database
            chunk = db.query(DocumentChunk).filter(
                DocumentChunk.chunk_id == chunk_id
            ).first()
            
            if not chunk:
                return None
            
            # Get document metadata
            document = db.query(Document).filter(Document.id == chunk.document_id).first()
            
            if not document:
                return None
            
            return SearchResult(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                text=chunk.text,
                score=vector_result.get('score', 0.0),
                metadata={
                    'chunk_index': chunk.chunk_index,
                    'text_length': chunk.text_length,
                    'start_char': chunk.start_char,
                    'end_char': chunk.end_char,
                    'embedding_model': chunk.embedding_model
                },
                document_metadata={
                    'filename': document.filename,
                    'content_type': document.content_type,
                    'created_at': document.created_at.isoformat(),
                    'version': document.version,
                    'title': document.title,
                    'tags': document.get_tag_list()  # Include document tags
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to create search result: {e}")
            return None
    
    async def _create_search_result_from_contextual(self, contextual_result: Dict, db: Session) -> Optional[SearchResult]:
        """Create SearchResult from contextual search result."""
        try:
            chunk_id = contextual_result.get('chunk_id')
            
            if not chunk_id:
                return None
            
            # Get chunk from database
            chunk = db.query(DocumentChunk).filter(
                DocumentChunk.chunk_id == chunk_id
            ).first()
            
            if not chunk:
                return None
            
            # Get document metadata
            document = db.query(Document).filter(Document.id == chunk.document_id).first()
            
            if not document:
                return None
            
            return SearchResult(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                text=chunk.text,
                score=contextual_result.get('score', 0.0),
                metadata={
                    'chunk_index': chunk.chunk_index,
                    'text_length': chunk.text_length,
                    'content_score': contextual_result.get('content_score', 0.0),
                    'context_score': contextual_result.get('context_score', 0.0),
                    'embedding_model': chunk.embedding_model
                },
                document_metadata={
                    'filename': document.filename,
                    'content_type': document.content_type,
                    'created_at': document.created_at.isoformat(),
                    'version': document.version,
                    'title': document.title,
                    'tags': document.get_tag_list()  # Include document tags
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to create contextual search result: {e}")
            return None
    
    def _generate_highlight(self, query: str, text: str, max_length: int = 200) -> str:
        """Generate highlighted text snippet."""
        try:
            query_terms = query.lower().split()
            text_lower = text.lower()
            
            # Find first match
            first_match_pos = None
            for term in query_terms:
                pos = text_lower.find(term)
                if pos != -1:
                    if first_match_pos is None or pos < first_match_pos:
                        first_match_pos = pos
            
            if first_match_pos is None:
                return text[:max_length] + "..." if len(text) > max_length else text
            
            # Extract snippet around first match
            start = max(0, first_match_pos - max_length // 2)
            end = min(len(text), start + max_length)
            
            snippet = text[start:end]
            
            # Add ellipsis if truncated
            if start > 0:
                snippet = "..." + snippet
            if end < len(text):
                snippet = snippet + "..."
            
            return snippet
            
        except Exception as e:
            logger.error(f"Failed to generate highlight: {e}")
            return text[:max_length]
    
    async def _get_cached_results(
        self,
        query: str,
        user_id: int,
        search_type: str,
        filters: SearchFilter,
        db: Session
    ) -> Optional[List[SearchResult]]:
        """Get cached search results if available."""
        try:
            import hashlib
            
            # Create query hash for cache lookup (include search type)
            cache_key = f"{query}_{user_id}_{search_type}_{filters.to_dict()}"
            query_hash = hashlib.sha256(cache_key.encode()).hexdigest()
            
            # Look for cached query
            cached_query = db.query(SearchQuery).filter(
                SearchQuery.query_hash == query_hash,
                SearchQuery.user_id == user_id
            ).first()
            
            if cached_query and cached_query.is_cache_valid():
                cached_results_data = cached_query.get_cached_results()
                
                # Convert back to SearchResult objects
                results = []
                for result_data in cached_results_data:
                    result = SearchResult(
                        chunk_id=result_data['chunk_id'],
                        document_id=result_data['document_id'],
                        text=result_data['text'],
                        score=result_data['score'],
                        metadata=result_data['metadata'],
                        document_metadata=result_data.get('document_metadata', {}),
                        highlight=result_data.get('highlight')
                    )
                    results.append(result)
                
                return results
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get cached results: {e}")
            return None
    
    async def _cache_results(
        self,
        query: str,
        user_id: int,
        search_type: str,
        filters: SearchFilter,
        results: List[SearchResult],
        db: Session
    ):
        """Cache search results for future use."""
        try:
            import hashlib
            
            # Create query hash (include search type)
            cache_key = f"{query}_{user_id}_{search_type}_{filters.to_dict()}"
            query_hash = hashlib.sha256(cache_key.encode()).hexdigest()
            
            # Serialize results
            results_data = [result.to_dict() for result in results]
            
            # Create or update search query record
            search_query = db.query(SearchQuery).filter(
                SearchQuery.query_hash == query_hash,
                SearchQuery.user_id == user_id
            ).first()
            
            if search_query:
                search_query.results_count = len(results)
                search_query.set_cached_results(results_data, self.cache_duration_hours)
            else:
                search_query = SearchQuery(
                    user_id=user_id,
                    query_text=query,
                    query_hash=query_hash,
                    query_type=search_type,
                    results_count=len(results)
                )
                search_query.set_cached_results(results_data, self.cache_duration_hours)
                db.add(search_query)
            
            db.commit()
            
        except Exception as e:
            logger.error(f"Failed to cache results: {e}")
    
    async def _log_search_query(
        self,
        query: str,
        user_id: int,
        search_type: str,
        filters: SearchFilter,
        results_count: int,
        search_time_ms: float,
        db: Session
    ):
        """Log search query for analytics."""
        try:
            import hashlib
            
            query_hash = hashlib.sha256(f"{query}_{user_id}".encode()).hexdigest()
            
            search_query = SearchQuery(
                user_id=user_id,
                query_text=query,
                query_hash=query_hash,
                query_type=search_type,
                search_filters=str(filters.to_dict()),
                results_count=results_count,
                search_time_ms=int(search_time_ms)
            )
            
            db.add(search_query)
            db.commit()
            
        except Exception as e:
            logger.error(f"Failed to log search query: {e}")
    
    async def _apply_reranking(
        self,
        query: str,
        results: List[SearchResult],
        filters: SearchFilter,
        search_type: str
    ) -> List[SearchResult]:
        """
        Apply reranking to search results.
        
        Args:
            query: Search query
            results: Initial search results
            filters: Search filters with reranker settings
            search_type: Type of search performed
            
        Returns:
            Reranked search results
        """
        if not results:
            return results
        
        try:
            # Check if reranking is enabled for this search type
            if not self._should_rerank_for_search_type(search_type):
                logger.debug(f"Reranking disabled for search type: {search_type}")
                return results
            
            # Limit results to rerank if too many
            initial_count = len(results)
            if initial_count > filters.max_results_to_rerank:
                results_to_rerank = results[:filters.max_results_to_rerank]
                remaining_results = results[filters.max_results_to_rerank:]
            else:
                results_to_rerank = results
                remaining_results = []
            
            # Check minimum threshold
            if len(results_to_rerank) < self.reranker_config.min_results_for_reranking:
                logger.debug(f"Too few results ({len(results_to_rerank)}) for reranking, minimum: {self.reranker_config.min_results_for_reranking}")
                return results
            
            # Convert to reranker format
            reranker_results = self._convert_to_reranker_format(results_to_rerank)
            
            # Apply reranking
            reranked = await self.reranker_manager.rerank(
                query=query,
                results=reranker_results,
                model_name=filters.reranker_model,
                top_k=None,  # Don't limit here, we'll combine with remaining results
                score_weight=filters.rerank_score_weight,
                min_rerank_score=filters.min_rerank_score
            )
            
            # Convert back to SearchResult format
            final_results = self._convert_from_reranker_format(reranked)
            
            # Add remaining results (if any) with their original scores
            final_results.extend(remaining_results)
            
            logger.info(f"Reranked {len(results_to_rerank)} results using {filters.reranker_model or 'default'} model")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Reranking failed, returning original results: {e}")
            return results
    
    def _should_rerank_for_search_type(self, search_type: str) -> bool:
        """Check if reranking should be applied for the given search type."""
        config = self.reranker_config
        
        if search_type == SearchType.SEMANTIC:
            return config.enable_for_semantic_search
        elif search_type == SearchType.HYBRID:
            return config.enable_for_hybrid_search
        elif search_type == SearchType.CONTEXTUAL:
            return config.enable_for_contextual_search
        
        # Default: enable for keyword search too
        return True
    
    def _convert_to_reranker_format(self, results: List[SearchResult]) -> List[RerankerSearchResult]:
        """Convert SearchResult objects to reranker format."""
        reranker_results = []
        
        for result in results:
            reranker_result = RerankerSearchResult(
                document_id=result.document_id,
                chunk_id=result.chunk_id,
                text=result.text,
                score=result.score,
                metadata=result.metadata,
                document_metadata=result.document_metadata,
                highlight=result.highlight
            )
            reranker_results.append(reranker_result)
        
        return reranker_results
    
    def _convert_from_reranker_format(self, reranked: List[RerankResult]) -> List[SearchResult]:
        """Convert reranked results back to SearchResult format."""
        search_results = []
        
        for rerank_result in reranked:
            # Update metadata to include reranking information
            updated_metadata = {
                **rerank_result.metadata,
                'original_score': rerank_result.original_score,
                'rerank_score': rerank_result.rerank_score,
                'combined_score': rerank_result.combined_score,
                'reranker_model': rerank_result.reranker_model
            }
            
            search_result = SearchResult(
                chunk_id=rerank_result.chunk_id,
                document_id=rerank_result.document_id,
                text=rerank_result.text,
                score=rerank_result.combined_score,  # Use combined score as the main score
                metadata=updated_metadata,
                document_metadata=rerank_result.document_metadata,
                highlight=rerank_result.highlight
            )
            
            search_results.append(search_result)
        
        return search_results

    async def get_search_suggestions(self, query: str, user: User, limit: int = 5, 
                                   db: Session = None) -> List[Dict[str, Any]]:
        """
        Generate search suggestions based on query, user's documents, and search history.
        """
        suggestions = []
        
        try:
            if not db:
                from database.connection import SessionLocal
                db = SessionLocal()
            
            # Search history suggestions
            recent_searches = db.query(SearchQuery).filter(
                SearchQuery.user_id == user.id,
                SearchQuery.query_text.ilike(f'%{query}%')
            ).order_by(SearchQuery.created_at.desc()).limit(3).all()
            
            for search in recent_searches:
                suggestions.append({
                    'type': 'history',
                    'text': search.query_text,
                    'icon': 'history'
                })
            
            # Document title suggestions
            documents = db.query(Document).filter(
                Document.user_id == user.id,
                Document.filename.ilike(f'%{query}%')
            ).limit(3).all()
            
            for doc in documents:
                suggestions.append({
                    'type': 'document_title',
                    'text': doc.filename.replace('.pdf', '').replace('.txt', ''),
                    'icon': 'document'
                })
            
            # Remove duplicates and limit results
            unique_suggestions = []
            seen_texts = set()
            for suggestion in suggestions:
                if suggestion['text'].lower() not in seen_texts:
                    seen_texts.add(suggestion['text'].lower())
                    unique_suggestions.append(suggestion)
                    if len(unique_suggestions) >= limit:
                        break
            
            return unique_suggestions
            
        except Exception as e:
            logger.error(f"Failed to get search suggestions: {e}")
            return []

    async def get_available_filters(self, user: User, db: Session = None) -> Dict[str, Any]:
        """
        Get available search filters based on user's document collection.
        """
        try:
            if not db:
                from database.connection import SessionLocal
                db = SessionLocal()
            
            # Get user's documents for filter options
            documents = db.query(Document).filter(Document.user_id == user.id).all()
            
            # Extract file types
            file_types = {}
            languages = set()
            date_range = {'min_date': None, 'max_date': None}
            file_sizes = []
            
            for doc in documents:
                # File types
                content_type = doc.content_type or 'unknown'
                if content_type not in file_types:
                    file_types[content_type] = 0
                file_types[content_type] += 1
                
                # Date range
                if doc.created_at:
                    if not date_range['min_date'] or doc.created_at < date_range['min_date']:
                        date_range['min_date'] = doc.created_at.isoformat()
                    if not date_range['max_date'] or doc.created_at > date_range['max_date']:
                        date_range['max_date'] = doc.created_at.isoformat()
                
                # File sizes
                if hasattr(doc, 'file_size') and doc.file_size:
                    file_sizes.append(doc.file_size)
            
            # Convert to API format
            file_type_options = [
                {
                    'value': file_type,
                    'label': file_type.split('/')[-1].upper(),
                    'count': count,
                    'icon': self._get_file_type_icon(file_type)
                }
                for file_type, count in file_types.items()
            ]
            
            # File size statistics
            file_size_range = {
                'min_size': min(file_sizes) if file_sizes else 0,
                'max_size': max(file_sizes) if file_sizes else 0,
                'avg_size': sum(file_sizes) // len(file_sizes) if file_sizes else 0
            }
            
            return {
                'file_types': file_type_options,
                'tags': await self._get_available_tags(user, db),
                'languages': await self._get_available_languages(user, db),
                'folders': await self._get_available_folders(user, db),
                'date_range': date_range,
                'file_size_range': file_size_range,
                'search_types': [
                    {'value': 'semantic', 'label': 'Semantic', 'description': 'AI-powered meaning-based search'},
                    {'value': 'keyword', 'label': 'Keyword', 'description': 'Exact word matching'},
                    {'value': 'hybrid', 'label': 'Hybrid', 'description': 'Best of both semantic and keyword'}
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to get available filters: {e}")
            return {
                'file_types': [],
                'tags': [],
                'languages': [],
                'folders': [],
                'date_range': {'min_date': None, 'max_date': None},
                'file_size_range': {'min_size': 0, 'max_size': 0, 'avg_size': 0},
                'search_types': []
            }

    async def _get_available_tags(self, user: User, db: Session) -> List[Dict[str, str]]:
        """Get list of available tags from documents accessible to the user."""
        try:
            # Build query for accessible documents with tags
            base_filter = and_(
                Document.is_deleted == False,
                Document.tags.is_not(None),  # Only documents with tags
                Document.tags != "",  # Exclude empty tag strings
                or_(
                    Document.user_id == user.id,  # User's own documents
                    Document.is_public == True,   # Public documents
                )
            )
            
            # Additional filter for admin users
            if user.has_role("admin"):
                # Admins can access all documents
                query = db.query(Document.tags).filter(
                    Document.is_deleted == False,
                    Document.tags.is_not(None),
                    Document.tags != ""
                )
            else:
                query = db.query(Document.tags).filter(base_filter)
            
            # Get all tag strings from accessible documents
            tag_records = query.all()
            
            # Parse and collect all unique tags
            all_tags = set()
            for (tag_string,) in tag_records:
                if tag_string:
                    try:
                        tags_list = json.loads(tag_string)
                        if isinstance(tags_list, list):
                            for tag in tags_list:
                                if tag and isinstance(tag, str):
                                    all_tags.add(tag.strip().lower())
                    except (json.JSONDecodeError, TypeError):
                        # Skip malformed tag data
                        continue
            
            # Convert to sorted list of dictionaries for frontend
            tag_options = [
                {
                    'value': tag,
                    'label': tag.title()  # Capitalize for display
                }
                for tag in sorted(all_tags)
            ]
            
            logger.debug(f"Found {len(tag_options)} unique tags for user {user.id}")
            return tag_options
            
        except Exception as e:
            logger.error(f"Failed to get available tags: {e}")
            return []

    async def _get_available_languages(self, user: User, db: Session) -> List[Dict[str, str]]:
        """Get list of available languages from documents accessible to the user."""
        try:
            # Build query for accessible documents with languages
            base_filter = and_(
                Document.is_deleted == False,
                Document.language.is_not(None),  # Only documents with language info
                Document.language != "",  # Exclude empty language strings
                or_(
                    Document.user_id == user.id,  # User's own documents
                    Document.is_public == True,   # Public documents
                )
            )
            
            # Additional filter for admin users
            if user.has_role("admin"):
                # Admins can access all documents
                query = db.query(Document.language).filter(
                    Document.is_deleted == False,
                    Document.language.is_not(None),
                    Document.language != ""
                ).distinct()
            else:
                query = db.query(Document.language).filter(base_filter).distinct()
            
            # Get unique language codes from accessible documents
            language_records = query.all()
            
            # Language code to name mapping (common languages)
            language_names = {
                'en': 'English',
                'es': 'Spanish',
                'fr': 'French',
                'de': 'German',
                'it': 'Italian',
                'pt': 'Portuguese',
                'ru': 'Russian',
                'ja': 'Japanese',
                'ko': 'Korean',
                'zh': 'Chinese',
                'ar': 'Arabic',
                'hi': 'Hindi',
                'nl': 'Dutch',
                'sv': 'Swedish',
                'no': 'Norwegian',
                'da': 'Danish',
                'fi': 'Finnish',
                'pl': 'Polish',
                'cs': 'Czech',
                'tr': 'Turkish'
            }
            
            # Convert to list of dictionaries for frontend
            language_options = []
            for (language_code,) in language_records:
                if language_code:
                    language_code = language_code.strip().lower()
                    language_name = language_names.get(language_code, language_code.upper())
                    language_options.append({
                        'value': language_code,
                        'label': language_name
                    })
            
            # Sort by language name for better UX
            language_options.sort(key=lambda x: x['label'])
            
            logger.debug(f"Found {len(language_options)} languages for user {user.id}")
            return language_options
            
        except Exception as e:
            logger.error(f"Failed to get available languages: {e}")
            return []

    async def _get_available_folders(self, user: User, db: Session) -> List[Dict[str, str]]:
        """Get list of available folder paths from documents accessible to the user."""
        try:
            # Build query for accessible documents with folder paths
            base_filter = and_(
                Document.is_deleted == False,
                Document.folder_path.is_not(None),  # Only documents with folder paths
                Document.folder_path != "",  # Exclude empty folder paths
                or_(
                    Document.user_id == user.id,  # User's own documents
                    Document.is_public == True,   # Public documents
                )
            )
            
            # Additional filter for admin users
            if user.has_role("admin"):
                # Admins can access all documents
                query = db.query(Document.folder_path).filter(
                    Document.is_deleted == False,
                    Document.folder_path.is_not(None),
                    Document.folder_path != ""
                ).distinct()
            else:
                query = db.query(Document.folder_path).filter(base_filter).distinct()
            
            # Get unique folder paths from accessible documents
            folder_records = query.all()
            
            # Process folder paths to create hierarchical structure
            folder_set = set()
            for (folder_path,) in folder_records:
                if folder_path:
                    # Normalize folder path (remove leading/trailing slashes)
                    normalized_path = folder_path.strip().strip('/')
                    if normalized_path:
                        # Add the full path
                        folder_set.add(normalized_path)
                        
                        # Also add parent paths for hierarchical navigation
                        path_parts = normalized_path.split('/')
                        for i in range(len(path_parts)):
                            parent_path = '/'.join(path_parts[:i+1])
                            if parent_path:
                                folder_set.add(parent_path)
            
            # Convert to list of dictionaries for frontend
            folder_options = []
            for folder_path in sorted(folder_set):
                # Create a display label that shows hierarchy
                display_name = folder_path.replace('/', ' / ')
                folder_options.append({
                    'value': folder_path,
                    'label': display_name,
                    'depth': folder_path.count('/')  # For frontend hierarchy display
                })
            
            # Sort by folder hierarchy (depth first, then alphabetically)
            folder_options.sort(key=lambda x: (x['depth'], x['label']))
            
            logger.debug(f"Found {len(folder_options)} folders for user {user.id}")
            return folder_options
            
        except Exception as e:
            logger.error(f"Failed to get available folders: {e}")
            return []

    def _get_file_type_icon(self, content_type: str) -> str:
        """Get appropriate icon for file type."""
        if 'pdf' in content_type:
            return 'file-pdf'
        elif 'image' in content_type:
            return 'file-image'
        elif 'text' in content_type:
            return 'file-text'
        elif 'doc' in content_type:
            return 'file-word'
        else:
            return 'file'

    async def get_recent_searches(self, user: User, limit: int = 10, 
                                db: Session = None) -> List[Dict[str, Any]]:
        """Get user's recent search queries."""
        try:
            if not db:
                from database.connection import SessionLocal
                db = SessionLocal()
            
            recent_searches = db.query(SearchQuery).filter(
                SearchQuery.user_id == user.id
            ).order_by(SearchQuery.created_at.desc()).limit(limit).all()
            
            return [
                {
                    'id': search.id,
                    'query': search.query,
                    'created_at': search.created_at.isoformat(),
                    'results_count': search.results_count or 0
                }
                for search in recent_searches
            ]
            
        except Exception as e:
            logger.error(f"Failed to get recent searches: {e}")
            return []

    async def save_search(self, user: User, query_data: Dict[str, Any], 
                         name: str, db: Session = None) -> Dict[str, Any]:
        """Save a search query for later use."""
        try:
            if not db:
                from database.connection import SessionLocal
                db = SessionLocal()
            
            from database.models import SavedSearch
            
            saved_search = SavedSearch(
                user_id=user.id,
                name=name,
                search_query=query_data,
                created_at=datetime.now(timezone.utc)
            )
            
            db.add(saved_search)
            db.commit()
            db.refresh(saved_search)
            
            return {
                'id': saved_search.id,
                'name': saved_search.name,
                'created_at': saved_search.created_at.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to save search: {e}")
            raise

    async def get_saved_searches(self, user: User, db: Session = None) -> List[Dict[str, Any]]:
        """Get user's saved searches."""
        try:
            if not db:
                from database.connection import SessionLocal
                db = SessionLocal()
            
            from database.models import SavedSearch
            
            saved_searches = db.query(SavedSearch).filter(
                SavedSearch.user_id == user.id
            ).order_by(SavedSearch.created_at.desc()).all()
            
            return [
                {
                    'id': search.id,
                    'name': search.name,
                    'search_query': search.search_query,
                    'created_at': search.created_at.isoformat()
                }
                for search in saved_searches
            ]
            
        except Exception as e:
            logger.error(f"Failed to get saved searches: {e}")
            return []

    async def invalidate_cache_for_documents(self, document_ids: List[int]):
        """Invalidate cache entries related to specific documents."""
        try:
            # Clear Redis cache entries containing these document IDs
            for doc_id in document_ids:
                pattern = f"*doc_{doc_id}*"
                await self.redis_manager.delete_pattern(pattern)
                
            logger.info(f"Cache invalidated for documents: {document_ids}")
        except Exception as e:
            logger.error(f"Failed to invalidate cache: {e}")

    async def clear_user_cache(self, user_id: int):
        """Clear all cached results for a specific user."""
        try:
            # Clear Redis cache entries for this user
            await self.redis_manager.delete_pattern(f"search_*_{user_id}_*")
            logger.info(f"Cleared cache for user {user_id}")
        except Exception as e:
            logger.error(f"Failed to clear user cache: {e}")

    async def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        try:
            # Get query optimizer statistics
            stats = {"query_count": 0, "cache_hit_rate": 0.0}  # Basic stats without QueryOptimizer
            
            # Add Redis health check
            redis_healthy = await self.redis_manager.health_check()
            
            return {
                **stats,
                'redis_healthy': redis_healthy,
                'cache_enabled': True,
                'cache_duration_hours': self.cache_duration_hours
            }
        except Exception as e:
            logger.error(f"Failed to get cache statistics: {e}")
            return {'error': str(e), 'cache_enabled': False}

    async def optimize_cache_performance(self):
        """Optimize cache performance based on usage patterns."""
        try:
            # Cache optimization now handled directly by Redis manager
            logger.info("Cache performance optimization completed")
        except Exception as e:
            logger.error(f"Cache optimization failed: {e}")

    def _generate_query_id(self, **kwargs) -> str:
        """Generate a unique query ID for caching."""
        import hashlib
        # Create a deterministic hash from the parameters
        params_str = json.dumps(kwargs, sort_keys=True)
        return hashlib.md5(params_str.encode()).hexdigest()

    async def _check_vector_cache_coverage(
        self, 
        query: str, 
        filters: SearchFilter, 
        expected_docs: List[int]
    ) -> float:
        """
        Check what percentage of expected vector search results are already cached.
        Returns coverage ratio between 0.0 and 1.0.
        """
        try:
            if not expected_docs:
                return 0.0
                
            # Generate embeddings for cache key generation (lightweight check)
            embedding_manager = self._get_embedding_manager_instance()
            embeddings = await embedding_manager.generate_embeddings([query])
            query_vector = embeddings[0] if embeddings else []
            
            cached_docs = 0
            total_docs = len(expected_docs)
            
            for doc_id in expected_docs:
                # Generate the same query_id that optimize_vector_search would use
                query_id = self._generate_query_id(
                    query_type="vector_similarity",
                    query_vector=str(query_vector[:5]),  # Use first 5 components for cache key
                    index_name=f"doc_{doc_id}",
                    top_k=50,  # Default check value
                    filters={'vector_type': 'content'},
                    user_id=filters.user_id
                )
                
                # Check if this vector search is cached
                cache_key = f"query_cache:{query_id}"
                cached_result = await self.redis_manager.get_value(cache_key)
                
                if cached_result is not None:
                    cached_docs += 1
            
            coverage = cached_docs / total_docs if total_docs > 0 else 0.0
            logger.debug(f"Vector cache coverage: {coverage:.2%} ({cached_docs}/{total_docs} docs)")
            return coverage
            
        except Exception as e:
            logger.warning(f"Failed to check vector cache coverage: {e}")
            return 0.0

    async def _assemble_from_vector_cache(
        self,
        query: str,
        filters: SearchFilter,
        limit: int,
        expected_docs: List[int]
    ) -> List[SearchResult]:
        """
        Assemble search results from cached vector search results.
        Used when vector cache coverage is high enough to skip full search.
        """
        try:
            embedding_manager = self._get_embedding_manager_instance()
            embeddings = await embedding_manager.generate_embeddings([query])
            query_vector = embeddings[0] if embeddings else []
            
            all_results = {}  # chunk_id -> SearchResult
            
            for doc_id in expected_docs:
                query_id = self._generate_query_id(
                    query_type="vector_similarity",
                    query_vector=str(query_vector[:5]),  # Use first 5 components for cache key
                    index_name=f"doc_{doc_id}",
                    top_k=limit * 2,
                    filters={'vector_type': 'content'},
                    user_id=filters.user_id
                )
                
                cache_key = f"query_cache:{query_id}"
                cached_result = await self.redis_manager.get_value(cache_key)
                
                if cached_result and 'results' in cached_result:
                    for result in cached_result['results']:
                        chunk_id = result.get('metadata', {}).get('chunk_id')
                        if chunk_id:
                            # Convert cached result to SearchResult object
                            search_result = SearchResult(
                                chunk_id=chunk_id,
                                document_id=result.get('metadata', {}).get('document_id', doc_id),
                                text=result.get('content', ''),
                                score=result.get('score', 0.0),
                                metadata=result.get('metadata', {}),
                                document_metadata=result.get('document_metadata', {})
                            )
                            
                            # Keep best score for each chunk
                            if chunk_id not in all_results or search_result.score > all_results[chunk_id].score:
                                all_results[chunk_id] = search_result
            
            # Sort by score and limit results
            sorted_results = sorted(all_results.values(), key=lambda x: x.score, reverse=True)
            final_results = sorted_results[:limit]
            
            logger.info(f"Assembled {len(final_results)} results from vector cache")
            return final_results
            
        except Exception as e:
            logger.error(f"Failed to assemble from vector cache: {e}")
            return []


def detect_optimal_search_type(query: str) -> str:
    """Intelligent search type selection based on query characteristics."""
    query = query.strip()
    words = query.split()
    
    # Very short queries → keyword search
    if len(words) <= 1:
        return SearchType.KEYWORD
    
    # Short exact phrases → keyword search
    if len(words) <= 2 and not any(q in query.lower() for q in ['what', 'how', 'why', 'when', 'where', 'who']):
        return SearchType.KEYWORD
    
    # Question-like queries → semantic search  
    if query.lower().startswith(('what', 'how', 'why', 'when', 'where', 'who', 'which', 'can', 'should', 'would', 'could')):
        return SearchType.SEMANTIC
    
    # Contains technical terms or specific phrases → hybrid search
    technical_indicators = ['api', 'function', 'method', 'class', 'error', 'code', 'algorithm', 'implementation']
    if any(term in query.lower() for term in technical_indicators):
        return SearchType.HYBRID
    
    # Complex queries (more than 5 words) → hybrid search
    if len(words) > 5:
        return SearchType.HYBRID
        
    # Default to semantic for best results with moderate complexity
    return SearchType.SEMANTIC


# Global search engine instance
_search_engine: Optional[EnhancedSearchEngine] = None

def get_search_engine() -> EnhancedSearchEngine:
    """Get the global enhanced search engine instance."""
    global _search_engine
    if _search_engine is None:
        # Use the optimized storage manager singleton
        storage_manager = get_storage_manager()
        _search_engine = EnhancedSearchEngine(storage_manager)
    return _search_engine

async def get_initialized_search_engine() -> EnhancedSearchEngine:
    """Get the global search engine instance and ensure storage is initialized."""
    search_engine = get_search_engine()
    if not search_engine._storage_initialized:
        await search_engine._ensure_storage_initialized()
    return search_engine