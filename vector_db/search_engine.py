"""
Enhanced search engine with user-aware access control and hybrid search capabilities.
Integrates FAISS, Qdrant, and database for comprehensive document search.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func

from database.models import Document, DocumentChunk, User, SearchQuery
from .storage_manager import VectorStorageManager, get_storage_manager
from .context_processor import ContextProcessor
from .reranker import get_reranker_manager, get_reranker_config, RerankResult
from .reranker.base_reranker import SearchResult as RerankerSearchResult
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

class SearchFilter:
    """Search filter class for structured filtering."""
    
    def __init__(self):
        self.user_id: Optional[int] = None
        self.document_ids: Optional[List[int]] = None
        self.content_types: Optional[List[str]] = None
        self.date_range: Optional[Tuple[datetime, datetime]] = None
        self.file_size_range: Optional[Tuple[int, int]] = None
        self.tags: Optional[List[str]] = None
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
            'language': self.language,
            'is_public': self.is_public,
            'min_score': self.min_score,
            'enable_reranking': self.enable_reranking,
            'reranker_model': self.reranker_model,
            'rerank_score_weight': self.rerank_score_weight,
            'min_rerank_score': self.min_rerank_score,
            'max_results_to_rerank': self.max_results_to_rerank
        }

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
        
        # Reranker components
        self.reranker_manager = get_reranker_manager()
        self.reranker_config = get_reranker_config()
        
        # Search configuration
        self.default_limit = 10
        self.max_limit = 100
        self.cache_duration_hours = 24
    
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
        
        try:
            # Check cache first
            if use_cache:
                cached_results = await self._get_cached_results(query, user.id, filters, db)
                if cached_results:
                    logger.info(f"Returning cached results for query: {query[:50]}...")
                    return cached_results
            
            # Apply user access control to filters
            filters.user_id = user.id
            accessible_doc_ids = await self._get_accessible_documents(user, db)
            
            if not accessible_doc_ids:
                logger.info(f"No accessible documents for user {user.id}")
                return []
            
            filters.document_ids = accessible_doc_ids
            
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
            
            if use_cache and results:
                await self._cache_results(query, user.id, filters, results, db)
            
            logger.info(f"Search completed in {search_time:.2f}ms, returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            return []
    
    async def _get_accessible_documents(self, user: User, db: Session) -> List[int]:
        """Get list of document IDs accessible to the user."""
        try:
            # Build query for accessible documents
            query = db.query(Document.id).filter(
                Document.is_deleted == False,
                or_(
                    Document.user_id == user.id,  # User's own documents
                    Document.is_public == True,   # Public documents
                    and_(
                        user.is_superuser == True  # Admin access
                    )
                )
            )
            
            # Additional filter for admin users
            if user.has_role("admin"):
                # Admins can access all documents
                query = db.query(Document.id).filter(Document.is_deleted == False)
            
            doc_ids = [row[0] for row in query.all()]
            logger.debug(f"User {user.id} has access to {len(doc_ids)} documents")
            
            return doc_ids
            
        except Exception as e:
            logger.error(f"Failed to get accessible documents: {e}")
            return []
    
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
                        # Search content vectors
                        results = await self.storage_manager.search_vectors(
                            index_name=index_name,
                            query_vector=query_vector,
                            vector_type="content",
                            limit=limit * 2,  # Get more results for fusion
                            score_threshold=filters.min_score or 0.1,
                            use_faiss=True
                        )
                        
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
                
                # Search content vectors
                results = await self.storage_manager.search_vectors(
                    index_name=index_name,
                    query_vector=query_vector,
                    vector_type="content",
                    limit=limit,
                    score_threshold=filters.min_score,
                    use_faiss=True  # Use FAISS for fast search
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
        Perform keyword-based search using database full-text search.
        """
        try:
            # Build database query for keyword search
            base_query = db.query(DocumentChunk, Document).join(
                Document, DocumentChunk.document_id == Document.id
            ).filter(
                Document.id.in_(filters.document_ids),
                Document.is_deleted == False
            )
            
            # Add text search filter
            search_terms = query.split()
            text_filters = []
            for term in search_terms:
                text_filters.append(DocumentChunk.text.ilike(f"%{term}%"))
            
            if text_filters:
                base_query = base_query.filter(or_(*text_filters))
            
            # Execute query
            results = base_query.limit(limit).all()
            
            # Convert to SearchResult objects
            search_results = []
            for chunk, document in results:
                # Calculate simple relevance score based on term frequency
                score = self._calculate_keyword_score(query, chunk.text)
                
                search_result = SearchResult(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    text=chunk.text,
                    score=score,
                    metadata={
                        'chunk_index': chunk.chunk_index,
                        'text_length': chunk.text_length,
                        'start_char': chunk.start_char,
                        'end_char': chunk.end_char
                    },
                    document_metadata={
                        'filename': document.filename,
                        'content_type': document.content_type,
                        'created_at': document.created_at.isoformat(),
                        'version': document.version
                    },
                    highlight=self._generate_highlight(query, chunk.text)
                )
                search_results.append(search_result)
            
            # Sort by score
            search_results.sort(key=lambda x: x.score, reverse=True)
            return search_results
            
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
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
                    'title': document.title
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
                    'title': document.title
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to create contextual search result: {e}")
            return None
    
    def _calculate_keyword_score(self, query: str, text: str) -> float:
        """
        Calculate simple keyword relevance score using custom algorithm (NOT BM25).
        
        Algorithm:
        1. Term Coverage (70% weight): Ratio of unique query terms found in text
        2. Frequency Score (30% weight): Term frequency relative to document length
        
        Formula: score = (term_coverage * 0.7) + (frequency_score * 0.3)
        
        Example:
        Query: "machine learning algorithm"
        Text: "Machine learning is a subset of AI. Learning algorithms are powerful."
        
        - unique_matches = 2 (machine, learning) 
        - total_matches = 3 (machine=1, learning=2, algorithm=0)
        - term_coverage = 2/3 = 0.67
        - frequency_score = min(3/14, 1.0) = 0.21
        - final_score = (0.67 * 0.7) + (0.21 * 0.3) = 0.53
        """
        try:
            query_terms = query.lower().split()
            text_lower = text.lower()
            
            if not query_terms:
                return 0.0
            
            # Count term frequencies
            total_matches = 0
            unique_matches = 0
            
            for term in query_terms:
                count = text_lower.count(term)
                if count > 0:
                    total_matches += count
                    unique_matches += 1
            
            # Calculate score based on term frequency and coverage
            term_coverage = unique_matches / len(query_terms)
            frequency_score = min(total_matches / len(text.split()), 1.0)
            
            return (term_coverage * 0.7) + (frequency_score * 0.3)
            
        except Exception as e:
            logger.error(f"Failed to calculate keyword score: {e}")
            return 0.0
    
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
        filters: SearchFilter,
        db: Session
    ) -> Optional[List[SearchResult]]:
        """Get cached search results if available."""
        try:
            import hashlib
            
            # Create query hash for cache lookup
            cache_key = f"{query}_{user_id}_{filters.to_dict()}"
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
        filters: SearchFilter,
        results: List[SearchResult],
        db: Session
    ):
        """Cache search results for future use."""
        try:
            import hashlib
            
            # Create query hash
            cache_key = f"{query}_{user_id}_{filters.to_dict()}"
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
                    query_type="semantic",
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
            
            # 1. Search history suggestions
            recent_searches = db.query(SearchQuery).filter(
                SearchQuery.user_id == user.id,
                SearchQuery.query.ilike(f'%{query}%')
            ).order_by(SearchQuery.created_at.desc()).limit(3).all()
            
            for search in recent_searches:
                suggestions.append({
                    'type': 'history',
                    'text': search.query,
                    'icon': 'history'
                })
            
            # 2. Document title suggestions
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
            
            # 3. Tag suggestions (if tags exist)
            # This would require tag implementation in the database
            
            # 4. Content-based suggestions using semantic search
            if len(suggestions) < limit and len(query) > 2:
                try:
                    search_results = await self.search(
                        query=query,
                        user=user,
                        search_type=SearchType.SEMANTIC,
                        limit=2,
                        db=db
                    )
                    
                    for result in search_results:
                        # Extract key phrases from the result text
                        words = result.text.split()[:10]  # First 10 words
                        suggestion_text = ' '.join(words)
                        suggestions.append({
                            'type': 'content',
                            'text': suggestion_text,
                            'icon': 'search'
                        })
                
                except Exception as e:
                    logger.debug(f"Content-based suggestions failed: {e}")
            
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
                'tags': [],  # TODO: Implement when tags are available
                'languages': [],  # TODO: Implement language detection
                'folders': [],  # TODO: Implement when folders are available
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
        _search_engine = EnhancedSearchEngine()
    return _search_engine