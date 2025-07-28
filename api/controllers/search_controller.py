from typing import List, Dict, Optional, Tuple
from fastapi import HTTPException
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
import hashlib
import json
import logging
from datetime import datetime

from vector_db.embedding_manager import EmbeddingManager
from vector_db.search_optimizer import SearchOptimizer, SearchError
from vector_db.context_processor import ContextProcessor
from vector_db.chunking import Chunk
from database.models import User, SearchQuery, Document, SavedSearch
from utils.security.audit_logger import AuditLogger

logger = logging.getLogger(__name__)

class SearchController:
    """Controller for all search operations."""
    
    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        content_search: SearchOptimizer,
        context_search: SearchOptimizer,
        context_processor: ContextProcessor,
        db: AsyncSession,
        audit_logger: AuditLogger
    ):
        self.embedding_manager = embedding_manager
        self.content_search = content_search
        self.context_search = context_search
        self.context_processor = context_processor
        self.db = db
        self.audit_logger = audit_logger

    async def search(
        self,
        query: str,
        user: User,
        query_context: Optional[str] = None,
        k: int = 5,
        filters: Optional[Dict] = None,
        content_weight: float = 0.7,
        context_weight: float = 0.3,
        min_score: float = 0.0,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> List[Dict]:
        """Unified search endpoint with all features."""
        start_time = datetime.utcnow()
        
        try:
            # Check for cached results first
            query_hash = SearchQuery.create_query_hash(query, filters)
            cached_query = await self._get_cached_search(query_hash, user.id)
            
            if cached_query and cached_query.is_cache_valid():
                self.audit_logger.log_event(
                    user_id=user.id,
                    action="search_cache_hit",
                    resource_type="search",
                    resource_id=str(cached_query.id),
                    ip_address=ip_address,
                    user_agent=user_agent
                )
                return cached_query.get_cached_results()
            
            # Process query context
            processed_context = self.context_processor.process_context(
                query, query_context
            )
            
            # Generate embeddings
            query_emb, context_emb = await self.embedding_manager.generate_query_embeddings(
                query, processed_context
            )
            
            # Add user access control filters
            user_filters = await self._add_user_access_filters(filters, user)
            
            # Search in both spaces
            content_results = self.content_search.search(
                query_emb, k=k, min_score=min_score
            )
            context_results = self.context_search.search(
                context_emb, k=k, min_score=min_score
            )
            
            # Combine and filter results with access control
            results = await self._combine_and_filter_results(
                content_results,
                context_results,
                content_weight,
                context_weight,
                user_filters,
                user
            )
            
            formatted_results = self._format_results(results[:k])
            
            # Cache results and log search
            search_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            await self._save_search_query(
                query, user, query_hash, formatted_results, 
                search_time, user_filters, ip_address, user_agent
            )
            
            return formatted_results
            
        except Exception as e:
            # Log search error
            self.audit_logger.log_event(
                user_id=user.id,
                action="search_error",
                resource_type="search",
                error_message=str(e),
                ip_address=ip_address,
                user_agent=user_agent,
                status="error"
            )
            
            raise HTTPException(
                status_code=500,
                detail=f"Search failed: {str(e)}"
            )

    async def batch_search(
        self,
        queries: List[str],
        user: User,
        contexts: Optional[List[str]] = None,
        k: int = 5,
        filters: Optional[Dict] = None,
        content_weight: float = 0.7,
        context_weight: float = 0.3,
        min_score: float = 0.0,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> List[List[Dict]]:
        """Unified batch search endpoint."""
        try:
            # Add user access control filters
            user_filters = await self._add_user_access_filters(filters, user)
            
            # Process all contexts
            processed_contexts = [
                self.context_processor.process_context(q, c)
                for q, c in zip(queries, contexts or [None] * len(queries))
            ]
            
            # Generate all embeddings
            query_embs = []
            context_embs = []
            for query, context in zip(queries, processed_contexts):
                q_emb, c_emb = await self.embedding_manager.generate_query_embeddings(
                    query, context
                )
                query_embs.append(q_emb)
                context_embs.append(c_emb)
            
            # Perform batch search
            content_results = self.content_search.batch_search(
                np.vstack(query_embs),
                k=k,
                min_score=min_score
            )
            context_results = self.context_search.batch_search(
                np.vstack(context_embs),
                k=k,
                min_score=min_score
            )
            
            # Process each query's results
            batch_results = []
            for i, (q_content, q_context) in enumerate(zip(content_results, context_results)):
                combined = await self._combine_and_filter_results(
                    q_content,
                    q_context,
                    content_weight,
                    context_weight,
                    user_filters,
                    user
                )
                batch_results.append(self._format_results(combined[:k]))
                
                # Log each search query
                query_hash = SearchQuery.create_query_hash(queries[i], user_filters)
                await self._save_search_query(
                    queries[i], user, query_hash, batch_results[-1], 
                    0, user_filters, ip_address, user_agent, is_batch=True
                )
            
            # Log batch search event
            self.audit_logger.log_event(
                user_id=user.id,
                action="batch_search",
                resource_type="search",
                details={"query_count": len(queries), "total_results": sum(len(r) for r in batch_results)},
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            return batch_results
            
        except Exception as e:
            # Log batch search error
            self.audit_logger.log_event(
                user_id=user.id,
                action="batch_search_error",
                resource_type="search",
                error_message=str(e),
                ip_address=ip_address,
                user_agent=user_agent,
                status="error"
            )
            
            raise HTTPException(
                status_code=500,
                detail=f"Batch search failed: {str(e)}"
            )

    async def _combine_and_filter_results(
        self,
        content_results: List[Tuple[Chunk, float]],
        context_results: List[Tuple[Chunk, float]],
        content_weight: float,
        context_weight: float,
        filters: Optional[Dict],
        user: User
    ) -> List[Tuple[Chunk, float]]:
        """Combine, filter, and rank search results with access control."""
        combined_scores = {}
        accessible_documents = await self._get_accessible_documents(user)
        
        for results, weight in [
            (content_results, content_weight),
            (context_results, context_weight)
        ]:
            for chunk, score in results:
                # Check document access permissions
                document_id = chunk.metadata.get('document_id')
                if document_id and document_id not in accessible_documents:
                    continue
                    
                if filters and not self._matches_filters(chunk.metadata, filters):
                    continue
                
                chunk_id = f"{chunk.start_idx}_{chunk.end_idx}"
                if chunk_id in combined_scores:
                    combined_scores[chunk_id]["score"] += score * weight
                else:
                    combined_scores[chunk_id] = {
                        "chunk": chunk,
                        "score": score * weight
                    }
        
        results = [
            (item["chunk"], item["score"])
            for item in combined_scores.values()
        ]
        return sorted(results, key=lambda x: x[1], reverse=True)

    def _matches_filters(self, metadata: Dict, filters: Dict) -> bool:
        """Check if metadata matches filter criteria."""
        for key, value in filters.items():
            if key not in metadata:
                return False
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            elif metadata[key] != value:
                return False
        return True

    def _format_results(self, results: List[Tuple[Chunk, float]]) -> List[Dict]:
        """Format search results for API response."""
        return [
            {
                "text": chunk.text,
                "context": chunk.context_text,
                "metadata": chunk.metadata,
                "score": float(score),
                "start_idx": chunk.start_idx,
                "end_idx": chunk.end_idx
            }
            for chunk, score in results
        ]
    
    async def _get_cached_search(self, query_hash: str, user_id: int) -> Optional[SearchQuery]:
        """Get cached search results if available."""
        try:
            from sqlalchemy import select
            result = await self.db.execute(
                select(SearchQuery).where(
                    SearchQuery.query_hash == query_hash,
                    SearchQuery.user_id == user_id
                )
            )
            return result.scalar_one_or_none()
        except Exception:
            return None
    
    async def _add_user_access_filters(self, filters: Optional[Dict], user: User) -> Dict:
        """Add user access control filters to search filters."""
        user_filters = filters.copy() if filters else {}
        
        # Add user-specific document filters unless user is admin
        if not user.is_admin:
            accessible_docs = await self._get_accessible_documents(user)
            user_filters['accessible_documents'] = accessible_docs
        
        return user_filters
    
    async def _get_accessible_documents(self, user: User) -> List[str]:
        """Get list of document IDs that user can access."""
        try:
            from sqlalchemy import select, or_
            
            # Query documents user can access
            query = select(Document.id).where(
                or_(
                    Document.user_id == user.id,  # Own documents
                    Document.is_public == True,    # Public documents
                    user.is_admin == True          # Admin can access all
                )
            )
            
            result = await self.db.execute(query)
            return [str(doc_id[0]) for doc_id in result.fetchall()]
        except Exception:
            # If query fails, return empty list (restrictive)
            return []
    
    async def _save_search_query(
        self, 
        query: str, 
        user: User, 
        query_hash: str, 
        results: List[Dict], 
        search_time: float,
        filters: Optional[Dict] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        is_batch: bool = False
    ):
        """Save search query for analytics and caching."""
        try:
            search_query = SearchQuery(
                user_id=user.id,
                query_text=query,
                query_hash=query_hash,
                query_type="semantic",  # Default type
                search_filters=json.dumps(filters) if filters else None,
                max_results=len(results),
                results_count=len(results),
                search_time_ms=int(search_time),
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            # Cache results if not batch search
            if not is_batch and results:
                search_query.set_cached_results(results)
            
            self.db.add(search_query)
            await self.db.commit()
            
            # Log search event
            self.audit_logger.log_event(
                user_id=user.id,
                action="search" if not is_batch else "batch_search_item",
                resource_type="search",
                resource_id=str(search_query.id),
                details={
                    "query": query[:100],  # Truncate long queries
                    "results_count": len(results),
                    "search_time_ms": int(search_time)
                },
                ip_address=ip_address,
                user_agent=user_agent
            )
            
        except Exception as e:
            # Don't fail search if logging fails
            self.audit_logger.log_event(
                user_id=user.id,
                action="search_logging_error",
                resource_type="search",
                error_message=str(e),
                ip_address=ip_address,
                user_agent=user_agent
            )

# Module-level functions for compatibility with routes
async def search_documents(query: str, filters=None, sort: Optional[str] = None, 
                         page: int = 1, page_size: int = 10, user_id: int = None):
    """Search documents with actual text matching and filtering."""
    from ..schemas.search_schemas import SearchFilters
    from database.connection import get_db
    from sqlalchemy import or_, and_
    import time
    
    # Start performance timing
    start_time = time.time()
    
    logger.info(f"Starting text search for user {user_id} with query: '{query}', page: {page}, page_size: {page_size}")
    
    try:
        # Handle filters parameter - could be SearchFilters object or dict or None
        if filters is None:
            filters_applied = SearchFilters()
        elif hasattr(filters, 'dict'):  # It's a SearchFilters object
            filters_applied = filters
        else:  # It's a dict
            try:
                filters_applied = SearchFilters(**filters)
            except:
                filters_applied = SearchFilters()
        
        logger.debug(f"Applied filters: {filters_applied}")
        
        # Get database session
        db = next(get_db())
        
        # Simple user lookup by ID
        from database.models import User, Document
        user = db.query(User).filter(User.id == user_id).first() if user_id else None
        
        if not user:
            logger.warning(f"User {user_id} not found for search")
            execution_time = (time.time() - start_time) * 1000
            return {
                "results": [],
                "total_hits": 0,
                "execution_time_ms": execution_time,
                "filters_applied": filters_applied,
                "query_vector_id": None,
                "query": query
            }
        
        logger.debug(f"Found user: {user.username} (ID: {user.id})")
        
        # Build base query for accessible documents
        base_query = db.query(Document).filter(
            and_(
                Document.is_deleted == False,
                Document.status == 'completed',
                or_(
                    Document.user_id == user.id,  # User's own documents
                    Document.is_public == True     # Public documents
                )
            )
        )
        
        logger.debug("Built base query for accessible documents")
        
        # Apply text search if query provided
        if query and query.strip():
            query_cleaned = query.strip()
            logger.info(f"Applying text search for query: '{query_cleaned}'")
            
            # Create text search conditions
            search_conditions = []
            
            # Search in extracted_text (main content) - case insensitive
            if query_cleaned:
                search_conditions.append(
                    Document.extracted_text.ilike(f"%{query_cleaned}%")
                )
                # Search in title
                search_conditions.append(
                    Document.title.ilike(f"%{query_cleaned}%")
                )
                # Search in filename
                search_conditions.append(
                    Document.filename.ilike(f"%{query_cleaned}%")
                )
                # Search in description
                search_conditions.append(
                    Document.description.ilike(f"%{query_cleaned}%")
                )
            
            if search_conditions:
                base_query = base_query.filter(or_(*search_conditions))
                logger.debug(f"Added {len(search_conditions)} text search conditions")
        
        # Apply additional filters
        filter_count = 0
        
        # File type filtering
        if hasattr(filters_applied, 'file_types') and filters_applied.file_types:
            file_type_conditions = []
            for file_type in filters_applied.file_types:
                file_type_conditions.append(Document.content_type.ilike(f"%{file_type}%"))
            if file_type_conditions:
                base_query = base_query.filter(or_(*file_type_conditions))
                filter_count += 1
                logger.debug(f"Added file type filter: {filters_applied.file_types}")
        
        # Tag filtering
        if hasattr(filters_applied, 'tag_ids') and filters_applied.tag_ids:
            tag_conditions = []
            for tag in filters_applied.tag_ids:
                # Search in JSON tags field
                tag_conditions.append(Document.tags.ilike(f"%{tag}%"))
            if tag_conditions:
                base_query = base_query.filter(or_(*tag_conditions))
                filter_count += 1
                logger.debug(f"Added tag filter: {filters_applied.tag_ids}")
        
        # Date range filtering
        if hasattr(filters_applied, 'date_range') and filters_applied.date_range:
            try:
                start_date, end_date = filters_applied.date_range
                if start_date:
                    base_query = base_query.filter(Document.created_at >= start_date)
                    filter_count += 1
                if end_date:
                    base_query = base_query.filter(Document.created_at <= end_date)
                    filter_count += 1
                logger.debug(f"Added date range filter: {start_date} to {end_date}")
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid date range filter: {e}")
        
        logger.info(f"Applied {filter_count} additional filters")
        
        # Get total count before pagination
        total_count = base_query.count()
        logger.info(f"Total matching documents: {total_count}")
        
        # Apply sorting
        if sort:
            if sort == 'created_at_desc':
                base_query = base_query.order_by(Document.created_at.desc())
            elif sort == 'created_at_asc':
                base_query = base_query.order_by(Document.created_at.asc())
            elif sort == 'filename_asc':
                base_query = base_query.order_by(Document.filename.asc())
            elif sort == 'filename_desc':
                base_query = base_query.order_by(Document.filename.desc())
            elif sort == 'file_size_desc':
                base_query = base_query.order_by(Document.file_size.desc())
            elif sort == 'file_size_asc':
                base_query = base_query.order_by(Document.file_size.asc())
            else:
                # Default to relevance (most recent first)
                base_query = base_query.order_by(Document.created_at.desc())
        else:
            # Default sorting - most recent first
            base_query = base_query.order_by(Document.created_at.desc())
        
        # Apply pagination
        skip = (page - 1) * page_size
        paginated_query = base_query.offset(skip).limit(page_size)
        
        logger.debug(f"Applied pagination: skip={skip}, limit={page_size}")
        
        # Execute query and get results
        documents = paginated_query.all()
        logger.info(f"Retrieved {len(documents)} documents from database")
        
        # Convert to search result format with relevance scoring
        search_results = []
        for doc in documents:
            # Calculate relevance score based on text matches
            score = _calculate_relevance_score(doc, query)
            
            # Extract content snippet with highlighting context
            content_snippet = _extract_content_snippet(doc, query)
            
            # Get tags safely
            try:
                tags_list = doc.get_tag_list()
            except Exception as e:
                logger.warning(f"Failed to get tags for document {doc.id}: {e}")
                tags_list = []
            
            search_result = {
                "document_id": str(doc.id),
                "filename": doc.filename,
                "content_snippet": content_snippet,
                "score": score,
                "metadata": {
                    "title": doc.title,
                    "content_type": doc.content_type,
                    "file_size": doc.file_size,
                    "tags": tags_list,
                    "created_at": doc.created_at.isoformat() if doc.created_at else None,
                    "language": doc.language,
                    "description": doc.description,
                    "upload_date": doc.created_at.isoformat() if doc.created_at else None,
                    "owner": user.username
                }
            }
            search_results.append(search_result)
        
        # Sort results by relevance score (highest first)
        search_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Calculate final execution time
        execution_time = (time.time() - start_time) * 1000
        
        logger.info(f"Text search completed in {execution_time:.2f}ms, found {len(search_results)} results")
        
        result = {
            "results": search_results,
            "total_hits": total_count,
            "execution_time_ms": execution_time,
            "filters_applied": filters_applied,
            "query_vector_id": None,
            "query": query,
            "processing_time": execution_time / 1000.0  # For frontend compatibility
        }
        
        logger.debug(f"Returning search result: {len(result['results'])} results, {result['total_hits']} total hits")
        return result
        
    except Exception as e:
        execution_time = (time.time() - start_time) * 1000
        logger.error(f"Text search failed after {execution_time:.2f}ms: {str(e)}", exc_info=True)
        
        # Return empty results on error
        return {
            "results": [],
            "total_hits": 0,
            "execution_time_ms": execution_time,
            "filters_applied": SearchFilters(),
            "query_vector_id": None,
            "query": query,
            "processing_time": execution_time / 1000.0
        }

def _calculate_relevance_score(document, query: str) -> float:
    """Calculate relevance score for a document based on text matches."""
    if not query or not query.strip():
        return 0.5  # Default score for no query
    
    query_lower = query.lower().strip()
    score = 0.0
    
    try:
        # Title matches (highest weight)
        if document.title and query_lower in document.title.lower():
            score += 0.4
            # Exact title match gets bonus
            if query_lower == document.title.lower():
                score += 0.2
        
        # Filename matches (high weight)
        if document.filename and query_lower in document.filename.lower():
            score += 0.3
        
        # Description matches (medium weight)
        if document.description and query_lower in document.description.lower():
            score += 0.2
        
        # Content matches (lower weight but can accumulate)
        if document.extracted_text:
            content_lower = document.extracted_text.lower()
            # Count occurrences in content
            match_count = content_lower.count(query_lower)
            if match_count > 0:
                # Diminishing returns for multiple matches
                content_score = min(0.3, match_count * 0.05)
                score += content_score
        
        # Tag matches (medium weight)
        try:
            tags = document.get_tag_list()
            for tag in tags:
                if query_lower in tag.lower():
                    score += 0.15
                    break  # Only count once
        except Exception:
            pass
        
        # Ensure score is between 0 and 1
        score = min(1.0, max(0.1, score))
        
    except Exception as e:
        logger.warning(f"Error calculating relevance score for document {document.id}: {e}")
        score = 0.1  # Minimal score for errors
    
    return score

def _extract_content_snippet(document, query: str, max_length: int = 300) -> str:
    """Extract a relevant content snippet with context around query matches."""
    if not document.extracted_text:
        # Fallback to title or description
        if document.title:
            return document.title[:max_length]
        elif document.description:
            return document.description[:max_length]
        else:
            return f"Document: {document.filename}"
    
    content = document.extracted_text
    
    # If no query, return beginning of content
    if not query or not query.strip():
        return content[:max_length] + ("..." if len(content) > max_length else "")
    
    query_lower = query.lower().strip()
    content_lower = content.lower()
    
    # Find the first occurrence of the query
    match_pos = content_lower.find(query_lower)
    
    if match_pos == -1:
        # No match found, return beginning
        return content[:max_length] + ("..." if len(content) > max_length else "")
    
    # Extract context around the match
    context_start = max(0, match_pos - 100)  # 100 chars before
    context_end = min(len(content), match_pos + len(query) + 200)  # 200 chars after
    
    snippet = content[context_start:context_end]
    
    # Add ellipsis if we're not at the beginning/end
    if context_start > 0:
        snippet = "..." + snippet
    if context_end < len(content):
        snippet = snippet + "..."
    
    # Truncate if still too long
    if len(snippet) > max_length:
        snippet = snippet[:max_length-3] + "..."
    
    return snippet

async def similarity_search(query_text: str, filters=None, top_k: int = 5, 
                          threshold: float = 0.0, user_id: int = None):
    """Perform semantic similarity search using the enhanced search engine."""
    try:
        from ..schemas.search_schemas import SearchFilters
        from vector_db.search_engine import EnhancedSearchEngine, SearchFilter, SearchType
        from vector_db.storage_manager import get_storage_manager
        from vector_db.embedding_manager import EnhancedEmbeddingManager
        from database.connection import get_db
        
        # Get database session and user
        db = next(get_db())
        user = db.query(User).filter(User.id == user_id).first() if user_id else None
        
        if not user:
            logger.warning(f"User {user_id} not found for similarity search")
            # Fall back to text search for anonymous users
            return await search_documents(
                query=query_text,
                filters=filters,
                page_size=top_k,
                user_id=user_id
            )
        
        # Handle filters parameter
        search_filters = None
        if filters:
            search_filter = SearchFilter()
            search_filter.user_id = user_id
            
            if hasattr(filters, 'dict'):
                filter_dict = filters.dict()
            elif isinstance(filters, dict):
                filter_dict = filters
            else:
                filter_dict = {}
            
            # Map filters to SearchFilter object
            if filter_dict.get('file_types'):
                search_filter.content_types = filter_dict['file_types']
            if filter_dict.get('tag_ids'):
                search_filter.tags = filter_dict['tag_ids']
            if filter_dict.get('date_range'):
                try:
                    start_date, end_date = filter_dict['date_range']
                    search_filter.date_range = (
                        datetime.fromisoformat(start_date) if isinstance(start_date, str) else start_date,
                        datetime.fromisoformat(end_date) if isinstance(end_date, str) else end_date
                    )
                except (ValueError, TypeError):
                    pass
            
            search_filter.min_score = threshold
            search_filters = search_filter
        
        # Initialize search engine components
        try:
            storage_manager = get_storage_manager()
            embedding_manager = EnhancedEmbeddingManager.create_default_manager()
            search_engine = EnhancedSearchEngine(storage_manager, embedding_manager)
            
            # Perform semantic search
            search_results = await search_engine.search(
                query=query_text,
                user=user,
                search_type=SearchType.SEMANTIC,
                filters=search_filters,
                limit=top_k,
                db=db
            )
            
            # Format results to match API schema
            formatted_results = []
            for result in search_results:
                # Get document info
                document = db.query(Document).filter(
                    Document.id == result.document_id
                ).first()
                
                if document:
                    formatted_result = {
                        "document_id": str(result.document_id),
                        "filename": document.filename,
                        "content_snippet": result.text[:300] + "..." if len(result.text) > 300 else result.text,
                        "score": result.score,
                        "metadata": {
                            "title": document.title,
                            "content_type": document.content_type,
                            "file_size": document.file_size,
                            "tags": document.get_tag_list(),
                            "created_at": document.created_at.isoformat(),
                            "chunk_id": getattr(result, 'chunk_id', None),
                            "highlight": getattr(result, 'highlight', None)
                        }
                    }
                    formatted_results.append(formatted_result)
            
            # Return in SearchResponse format
            response = {
                "results": formatted_results,
                "total_hits": len(formatted_results),
                "execution_time_ms": 50.0,  # Would be measured in real implementation
                "filters_applied": SearchFilters(**(filters.dict() if hasattr(filters, 'dict') else filters or {})),
                "query_vector_id": None
            }
            
            logger.info(f"Semantic search completed for user {user_id}, found {len(formatted_results)} results")
            return response
            
        except Exception as search_error:
            logger.error(f"Vector search failed, falling back to text search: {search_error}")
            # Fall back to text search if vector search fails
            return await search_documents(
                query=query_text,
                filters=filters,
                page_size=top_k,
                user_id=user_id
            )
        
    except Exception as e:
        logger.error(f"Similarity search failed: {str(e)}")
        # Ultimate fallback to text search
        return await search_documents(
            query=query_text,
            filters=filters,
            page_size=top_k,
            user_id=user_id
        )

def _safe_get_tag_list(doc):
    """Safely get tag list from document with error handling."""
    try:
        return doc.get_tag_list()
    except Exception as e:
        logger.warning(f"Failed to get tag list for document {doc.id}: {e}")
        return []

async def get_available_filters(user_id: int):
    """Get available search filters."""
    logger.info(f"Getting available filters for user {user_id}")
    try:
        from database.connection import get_db
        from sqlalchemy import distinct, func
        
        db = next(get_db())
        logger.debug(f"Database session obtained for user {user_id}")
        
        # Get user's documents to extract available filter options
        user_documents = db.query(Document).filter(
            Document.user_id == user_id,
            Document.is_deleted == False
        ).all()
        
        # Extract unique file types
        file_types = set()
        tags = set()
        languages = set()
        folders = set()
        
        for doc in user_documents:
            # File types
            if doc.content_type:
                file_types.add(doc.content_type)
            
            # Tags - use the model method which has proper error handling
            try:
                doc_tags = doc.get_tag_list()
                if doc_tags:
                    tags.update(doc_tags)
            except Exception as e:
                logger.warning(f"Failed to extract tags from document {doc.id}: {e}")
                pass
            
            # Languages
            if doc.language:
                languages.add(doc.language)
            
            # Folders
            if doc.folder_path:
                folders.add(doc.folder_path)
        
        # Get date range of documents
        date_stats = db.query(
            func.min(Document.created_at),
            func.max(Document.created_at)
        ).filter(
            Document.user_id == user_id,
            Document.is_deleted == False
        ).first()
        
        min_date = date_stats[0].isoformat() if date_stats[0] else None
        max_date = date_stats[1].isoformat() if date_stats[1] else None
        
        # Get file size range
        size_stats = db.query(
            func.min(Document.file_size),
            func.max(Document.file_size),
            func.avg(Document.file_size)
        ).filter(
            Document.user_id == user_id,
            Document.is_deleted == False
        ).first()
        
        min_size = size_stats[0] or 0
        max_size = size_stats[1] or 0
        avg_size = int(size_stats[2]) if size_stats[2] else 0
        
        # Format available filters
        available_filters = {
            "file_types": [
                {
                    "value": ft,
                    "label": _get_file_type_label(ft),
                    "icon": _get_file_type_icon(ft)
                }
                for ft in sorted(file_types)
            ],
            "tags": [
                {
                    "value": tag,
                    "label": tag,
                    "count": sum(1 for doc in user_documents if tag in _safe_get_tag_list(doc))
                }
                for tag in sorted(tags) if tag and isinstance(tag, str)
            ],
            "languages": [
                {
                    "value": lang,
                    "label": _get_language_label(lang)
                }
                for lang in sorted(languages) if lang
            ],
            "folders": [
                {
                    "value": folder,
                    "label": folder,
                    "count": sum(1 for doc in user_documents if doc.folder_path == folder)
                }
                for folder in sorted(folders)
            ],
            "date_range": {
                "min_date": min_date,
                "max_date": max_date
            },
            "file_size_range": {
                "min_size": min_size,
                "max_size": max_size,
                "avg_size": avg_size
            },
            "search_types": [
                {"value": "basic", "label": "Basic Text Search", "description": "Keyword-based search"},
                {"value": "semantic", "label": "Semantic Search", "description": "AI-powered context understanding"},
                {"value": "hybrid", "label": "Hybrid Search", "description": "Combined text and semantic search"}
            ]
        }
        
        logger.info(f"Successfully generated available filters for user {user_id}: {len(available_filters['file_types'])} file types, {len(available_filters['tags'])} tags, {len(available_filters['languages'])} languages")
        return available_filters
        
    except Exception as e:
        logger.error(f"Failed to get available filters for user {user_id}: {str(e)}", exc_info=True)
        return {
            "file_types": [],
            "tags": [],
            "languages": [],
            "folders": [],
            "date_range": {"min_date": None, "max_date": None},
            "file_size_range": {"min_size": 0, "max_size": 0, "avg_size": 0},
            "search_types": [
                {"value": "basic", "label": "Basic Text Search", "description": "Keyword-based search"},
                {"value": "semantic", "label": "Semantic Search", "description": "AI-powered context understanding"},
                {"value": "hybrid", "label": "Hybrid Search", "description": "Combined text and semantic search"}
            ]
        }

def _get_file_type_label(content_type: str) -> str:
    """Get human-readable label for file type."""
    type_labels = {
        "application/pdf": "PDF Documents",
        "text/plain": "Text Files",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "Word Documents",
        "application/msword": "Word Documents (Legacy)",
        "text/html": "HTML Files",
        "application/json": "JSON Files",
        "text/markdown": "Markdown Files",
        "text/csv": "CSV Files",
        "image/png": "PNG Images",
        "image/jpeg": "JPEG Images",
        "image/gif": "GIF Images"
    }
    return type_labels.get(content_type, content_type)

def _get_file_type_icon(content_type: str) -> str:
    """Get icon name for file type."""
    type_icons = {
        "application/pdf": "document-text",
        "text/plain": "document",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "document",
        "application/msword": "document",
        "text/html": "code-bracket",
        "application/json": "code-bracket",
        "text/markdown": "document-text",
        "text/csv": "table-cells",
        "image/png": "photo",
        "image/jpeg": "photo",
        "image/gif": "photo"
    }
    return type_icons.get(content_type, "document")

def _get_language_label(language_code: str) -> str:
    """Get human-readable label for language code."""
    language_labels = {
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "pt": "Portuguese",
        "ja": "Japanese",
        "ko": "Korean",
        "zh": "Chinese",
        "ru": "Russian",
        "ar": "Arabic"
    }
    return language_labels.get(language_code, language_code.upper())

async def save_search(name: str, search_request: Dict, user_id: int):
    """Save a search query."""
    logger.info(f"Saving search '{name}' for user {user_id}")
    try:
        from database.connection import get_db
        
        db = next(get_db())
        
        # Extract search parameters from request
        query_text = search_request.get("query", "")
        search_type = search_request.get("search_type", "semantic")
        search_filters = search_request.get("filters", {})
        max_results = search_request.get("top_k", 10)
        similarity_threshold = search_request.get("similarity_threshold")
        
        # Validate input
        if not name or not name.strip():
            raise ValueError("Search name is required")
        if not query_text or not query_text.strip():
            raise ValueError("Search query is required")
        
        # Check if user already has a saved search with this name
        existing_search = db.query(SavedSearch).filter(
            SavedSearch.user_id == user_id,
            SavedSearch.name == name.strip(),
            SavedSearch.is_deleted == False
        ).first()
        
        if existing_search:
            raise ValueError(f"Saved search with name '{name}' already exists")
        
        # Create new saved search
        saved_search = SavedSearch(
            user_id=user_id,
            name=name.strip(),
            query_text=query_text.strip(),
            search_type=search_type,
            max_results=max_results,
            similarity_threshold=similarity_threshold
        )
        
        # Set filters if provided
        if search_filters:
            saved_search.set_filters(search_filters)
        
        db.add(saved_search)
        db.commit()
        db.refresh(saved_search)
        
        logger.info(f"Successfully saved search '{name}' (ID: {saved_search.id}) for user {user_id}")
        
        return saved_search
        
    except Exception as e:
        logger.error(f"Failed to save search '{name}' for user {user_id}: {str(e)}", exc_info=True)
        if "db" in locals():
            db.rollback()
        raise HTTPException(status_code=400, detail=str(e))

async def get_saved_searches(user_id: int):
    """Get user's saved searches."""
    logger.info(f"Retrieving saved searches for user {user_id}")
    try:
        from database.connection import get_db
        from sqlalchemy import desc
        
        db = next(get_db())
        
        # Get saved searches for the user
        saved_searches = db.query(SavedSearch).filter(
            SavedSearch.user_id == user_id,
            SavedSearch.is_deleted == False
        ).order_by(desc(SavedSearch.usage_count), desc(SavedSearch.created_at)).all()
        
        # Format the results
        formatted_searches = []
        for search in saved_searches:
            formatted_search = {
                "id": search.id,
                "name": search.name,
                "description": search.description,
                "query_text": search.query_text,
                "search_type": search.search_type,
                "filters": search.get_filters_dict(),
                "max_results": search.max_results,
                "similarity_threshold": search.similarity_threshold,
                "usage_count": search.usage_count,
                "created_at": search.created_at.isoformat(),
                "last_used": search.last_used.isoformat() if search.last_used else None,
                "tags": search.get_tags_list(),
                "is_public": search.is_public
            }
            formatted_searches.append(formatted_search)
        
        logger.info(f"Successfully retrieved {len(formatted_searches)} saved searches for user {user_id}")
        return formatted_searches
        
    except Exception as e:
        logger.error(f"Failed to get saved searches for user {user_id}: {str(e)}", exc_info=True)
        return []

async def get_recent_searches(user_id: int, limit: int = 10):
    """Get user's recent searches."""
    logger.info(f"Retrieving {limit} recent searches for user {user_id}")
    try:
        from database.connection import get_db
        from sqlalchemy import desc
        
        db = next(get_db())
        
        # Get recent search queries for the user
        recent_searches = db.query(SearchQuery).filter(
            SearchQuery.user_id == user_id
        ).order_by(desc(SearchQuery.created_at)).limit(limit).all()
        
        # Format the results
        formatted_searches = []
        for search in recent_searches:
            formatted_search = {
                "id": search.id,
                "query_text": search.query_text,
                "query_type": search.query_type,
                "created_at": search.created_at.isoformat(),
                "results_count": search.results_count,
                "search_time_ms": search.search_time_ms,
                "filters": search.get_filters_dict() if hasattr(search, 'get_filters_dict') else {}
            }
            
            # Add filters if available
            if search.search_filters:
                try:
                    import json
                    formatted_search["filters"] = json.loads(search.search_filters)
                except json.JSONDecodeError:
                    formatted_search["filters"] = {}
            
            formatted_searches.append(formatted_search)
        
        logger.info(f"Successfully retrieved {len(formatted_searches)} recent searches for user {user_id}")
        return formatted_searches
        
    except Exception as e:
        logger.error(f"Failed to get recent searches for user {user_id}: {str(e)}", exc_info=True)
        return []

async def get_search_suggestions(query: str, limit: int = 5, user_id: int = None):
    """Get search suggestions."""
    logger.debug(f"Getting search suggestions for query '{query}' (limit: {limit}, user: {user_id})")
    try:
        from database.connection import get_db
        from sqlalchemy import func, or_
        
        if not query or len(query.strip()) < 2:
            return []
        
        db = next(get_db())
        query_lower = query.lower().strip()
        
        suggestions = []
        
        # 1. Get suggestions from user's search history
        if user_id:
            recent_searches = db.query(SearchQuery.query_text).filter(
                SearchQuery.user_id == user_id,
                func.lower(SearchQuery.query_text).like(f"%{query_lower}%")
            ).distinct().limit(limit).all()
            
            for search in recent_searches:
                if search.query_text.lower() != query_lower:  # Don't suggest exact matches
                    suggestions.append({
                        "type": "history",
                        "text": search.query_text,
                        "icon": "clock"
                    })
        
        # 2. Get suggestions from document titles and content
        if user_id and len(suggestions) < limit:
            remaining_limit = limit - len(suggestions)
            
            # Search in document titles
            title_matches = db.query(Document.title).filter(
                Document.user_id == user_id,
                Document.is_deleted == False,
                Document.title.isnot(None),
                func.lower(Document.title).like(f"%{query_lower}%")
            ).distinct().limit(remaining_limit).all()
            
            for doc in title_matches:
                if doc.title and len(suggestions) < limit:
                    suggestions.append({
                        "type": "document_title",
                        "text": doc.title,
                        "icon": "document-text"
                    })
        
        # 3. Get suggestions from document tags
        if user_id and len(suggestions) < limit:
            remaining_limit = limit - len(suggestions)
            
            documents_with_tags = db.query(Document).filter(
                Document.user_id == user_id,
                Document.is_deleted == False,
                Document.tags.isnot(None)
            ).all()
            
            matching_tags = set()
            for doc in documents_with_tags:
                try:
                    tags = doc.get_tag_list()
                    for tag in tags:
                        if tag and isinstance(tag, str) and query_lower in tag.lower():
                            if len(matching_tags) < remaining_limit:
                                matching_tags.add(tag)
                except Exception as e:
                    logger.warning(f"Failed to extract tags for suggestions from document {doc.id}: {e}")
                    continue
            
            for tag in matching_tags:
                if len(suggestions) < limit:
                    suggestions.append({
                        "type": "tag",
                        "text": tag,
                        "icon": "tag"
                    })
        
        # 4. Get suggestions from saved searches
        if user_id and len(suggestions) < limit:
            remaining_limit = limit - len(suggestions)
            
            saved_searches = db.query(SavedSearch.name, SavedSearch.query_text).filter(
                SavedSearch.user_id == user_id,
                SavedSearch.is_deleted == False,
                or_(
                    func.lower(SavedSearch.name).like(f"%{query_lower}%"),
                    func.lower(SavedSearch.query_text).like(f"%{query_lower}%")
                )
            ).limit(remaining_limit).all()
            
            for search in saved_searches:
                if len(suggestions) < limit:
                    # Prefer the name if it matches, otherwise use the query
                    suggestion_text = search.name if query_lower in search.name.lower() else search.query_text
                    suggestions.append({
                        "type": "saved_search",
                        "text": suggestion_text,
                        "icon": "bookmark"
                    })
        
        # 5. Add generic search suggestions based on query patterns
        if len(suggestions) < limit:
            generic_suggestions = _get_generic_suggestions(query_lower)
            for suggestion in generic_suggestions:
                if len(suggestions) < limit:
                    suggestions.append(suggestion)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_suggestions = []
        for suggestion in suggestions:
            key = suggestion["text"].lower()
            if key not in seen:
                seen.add(key)
                unique_suggestions.append(suggestion)
        
        logger.debug(f"Generated {len(unique_suggestions)} unique suggestions for query '{query}'")
        return unique_suggestions[:limit]
        
    except Exception as e:
        logger.error(f"Failed to get search suggestions for query '{query}': {str(e)}", exc_info=True)
        return []

def _get_generic_suggestions(query: str) -> List[Dict[str, str]]:
    """Get generic search suggestions based on query patterns."""
    suggestions = []
    
    # Common search patterns
    patterns = {
        "how": ["how to", "how does", "how can"],
        "what": ["what is", "what are", "what does"],
        "why": ["why does", "why is", "why do"],
        "when": ["when to", "when is", "when does"],
        "where": ["where is", "where are", "where to"],
        "who": ["who is", "who are", "who can"]
    }
    
    for key, variations in patterns.items():
        if query.startswith(key) and len(query) > len(key):
            for variation in variations:
                if variation.startswith(query) and variation != query:
                    suggestions.append({
                        "type": "suggestion",
                        "text": variation,
                        "icon": "light-bulb"
                    })
                    break
    
    # Common document-related searches
    doc_searches = [
        "recent documents",
        "large files", 
        "pdf documents",
        "images",
        "shared documents",
        "untagged documents"
    ]
    
    for search in doc_searches:
        if query in search and len(suggestions) < 3:
            suggestions.append({
                "type": "suggestion", 
                "text": search,
                "icon": "magnifying-glass"
            })
    
    return suggestions