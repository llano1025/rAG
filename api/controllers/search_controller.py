from typing import List, Dict, Optional, Tuple
from fastapi import HTTPException
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
import hashlib
import json
from datetime import datetime

from vector_db.embedding_manager import EmbeddingManager
from vector_db.search_optimizer import SearchOptimizer, SearchError
from vector_db.context_processor import ContextProcessor
from vector_db.chunking import Chunk
from database.models import User, SearchQuery, Document
from utils.security.audit_logger import AuditLogger

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
                await self.audit_logger.log_event(
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
            await self.audit_logger.log_event(
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
            await self.audit_logger.log_event(
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
            await self.audit_logger.log_event(
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
            await self.audit_logger.log_event(
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
            await self.audit_logger.log_event(
                user_id=user.id,
                action="search_logging_error",
                resource_type="search",
                error_message=str(e),
                ip_address=ip_address,
                user_agent=user_agent
            )

# Module-level functions for compatibility with routes
async def search_documents(query: str, filters: Optional[Dict] = None, sort: Optional[str] = None, 
                         page: int = 1, page_size: int = 10, user_id: int = None):
    """Search documents with text and filtering."""
    # This is a placeholder - implement actual search logic
    return {
        "results": [],
        "total": 0,
        "page": page,
        "page_size": page_size,
        "query": query
    }

async def similarity_search(query_text: str, filters: Optional[Dict] = None, top_k: int = 5, 
                          threshold: float = 0.0, user_id: int = None):
    """Perform semantic similarity search."""
    # This is a placeholder - implement actual similarity search logic
    return {
        "results": [],
        "total": 0,
        "query": query_text
    }

async def get_available_filters(user_id: int):
    """Get available search filters."""
    return []

async def save_search(name: str, search_request: Dict, user_id: int):
    """Save a search query."""
    # Placeholder search object
    class SavedSearch:
        def __init__(self):
            self.id = 1
    return SavedSearch()

async def get_saved_searches(user_id: int):
    """Get user's saved searches."""
    return []

async def get_recent_searches(user_id: int, limit: int = 10):
    """Get user's recent searches."""
    return []

async def get_search_suggestions(query: str, limit: int = 5, user_id: int = None):
    """Get search suggestions."""
    return []