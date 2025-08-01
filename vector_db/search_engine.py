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
            'min_score': self.min_score
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
        use_cache: bool = True
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
            
            # Perform search using existing method
            results = await self.search(
                query=query,
                user=user,
                search_type=search_type,
                filters=None,
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
            elif search_type == SearchType.KEYWORD:
                results = await self._keyword_search(query, filters, limit, db)
            elif search_type == SearchType.HYBRID:
                results = await self._hybrid_search(query, filters, limit, db)
            elif search_type == SearchType.CONTEXTUAL:
                results = await self._contextual_search(query, filters, limit, db)
            else:
                raise ValueError(f"Unknown search type: {search_type}")
            
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
        """Perform keyword-based search using database full-text search."""
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
            
            # Calculate hybrid scores
            final_results = []
            semantic_weight = 0.7
            keyword_weight = 0.3
            
            for key, data in combined_results.items():
                hybrid_score = (
                    data['semantic_score'] * semantic_weight +
                    data['keyword_score'] * keyword_weight
                )
                
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
        """Calculate simple keyword relevance score."""
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


# Global search engine instance
_search_engine: Optional[EnhancedSearchEngine] = None

def get_search_engine() -> EnhancedSearchEngine:
    """Get the global enhanced search engine instance."""
    global _search_engine
    if _search_engine is None:
        _search_engine = EnhancedSearchEngine()
    return _search_engine