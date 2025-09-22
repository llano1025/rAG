from typing import List, Dict, Optional, Tuple
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
import json
import logging
from datetime import datetime, timezone

from vector_db.embedding_manager import EmbeddingManager
from vector_db.search_optimizer import SearchOptimizer, SearchError
from vector_db.search_context_processor import ContextProcessor
from vector_db.chunking import Chunk
from database.models import User, SearchQuery, Document, SavedSearch
from utils.security.audit_logger import AuditLogger
from utils.exceptions import (
    InvalidSearchQueryException,
    SearchEngineUnavailableException,
)

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
        embedding_model: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> List[Dict]:
        """Unified search endpoint with all features."""
        start_time = datetime.now(timezone.utc)

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

            # Generate embeddings with specified model or default
            if embedding_model:
                # Create temporary embedding manager with specified model
                from vector_db.embedding_manager import EnhancedEmbeddingManager
                temp_manager = EnhancedEmbeddingManager.create_default_manager(
                    model_name=embedding_model
                )
                query_emb, context_emb = await temp_manager.generate_query_embeddings(
                    query, processed_context
                )
            else:
                # Use default embedding manager
                query_emb, context_emb = await self.embedding_manager.generate_query_embeddings(
                    query, processed_context
                )

            # Add user access control filters
            user_filters = await self._add_user_access_filters(filters, user)

            # Search in both semantic spaces
            content_results = self.content_search.search(
                query_emb, k=k * 2, min_score=min_score  # Get more results for fusion
            )
            context_results = self.context_search.search(
                context_emb, k=k * 2, min_score=min_score
            )

            # Combine semantic results (enhanced search engine handles advanced fusion)
            results = await self._combine_and_filter_results(
                content_results,
                context_results,
                content_weight,
                context_weight,
                user_filters,
                user,
                min_score
            )

            formatted_results = self._format_results(results[:k])

            # Cache results and log search
            search_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
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

            if isinstance(e, SearchError):
                raise SearchEngineUnavailableException(
                    message=f"Search engine error: {str(e)}",
                    query=query
                )
            else:
                raise SearchEngineUnavailableException(
                    message=f"Search operation failed: {str(e)}",
                    query=query
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
                    user,
                    min_score
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

            raise SearchEngineUnavailableException(
                message=f"Batch search operation failed: {str(e)}",
                context={"batch_size": len(queries)}
            )

    async def _combine_and_filter_results(
        self,
        content_results: List[Tuple[Chunk, float]],
        context_results: List[Tuple[Chunk, float]],
        content_weight: float,
        context_weight: float,
        filters: Optional[Dict],
        user: User,
        min_score: float = 0.0
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
            if item["score"] >= min_score  # Apply final min_score filtering
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
                query_type="contextual",  # Default to contextual search
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

    async def _get_chunk_by_id(self, chunk_id: str):
        """Retrieve a chunk by its ID from the database."""
        try:
            from sqlalchemy import select
            from database.models import DocumentChunk
            from vector_db.chunking import Chunk

            result = await self.db.execute(
                select(DocumentChunk).where(DocumentChunk.chunk_id == chunk_id)
            )
            db_chunk = result.scalar_one_or_none()

            if db_chunk:
                # Convert database chunk to Chunk object
                return Chunk(
                    text=db_chunk.text,
                    start_idx=db_chunk.start_char,
                    end_idx=db_chunk.end_char,
                    metadata={'document_id': db_chunk.document_id, 'chunk_index': db_chunk.chunk_index},
                    context_text=f"{db_chunk.context_before} {db_chunk.context_after}".strip(),
                    document_id=db_chunk.document_id,
                    chunk_id=db_chunk.chunk_id
                )
            return None

        except Exception as e:
            logger.error(f"Failed to retrieve chunk {chunk_id}: {e}")
            return None


# Helper functions for standalone use (kept for backward compatibility)
async def save_search(name: str, search_request: Dict, user_id: int):
    """Save a search configuration."""
    try:
        from database.connection import get_db

        db = next(get_db())

        # Create saved search
        saved_search = SavedSearch(
            user_id=user_id,
            name=name,
            query_text=search_request.get('query', ''),
            search_type=search_request.get('search_type', 'contextual'),
            search_filters=json.dumps(search_request.get('filters', {})),
            max_results=search_request.get('max_results', 10),
            similarity_threshold=search_request.get('similarity_threshold')
        )

        db.add(saved_search)
        db.commit()

        logger.info(f"Saved search '{name}' for user {user_id}")

        return {
            "id": str(saved_search.id),
            "name": saved_search.name,
            "message": f"Search '{name}' saved successfully"
        }

    except Exception as e:
        logger.error(f"Failed to save search for user {user_id}: {str(e)}", exc_info=True)
        return {"error": f"Failed to save search: {str(e)}"}


async def get_saved_searches(user_id: int):
    """Get user's saved searches."""
    try:
        from database.connection import get_db

        db = next(get_db())

        saved_searches = db.query(SavedSearch).filter(
            SavedSearch.user_id == user_id,
            SavedSearch.is_deleted == False
        ).order_by(SavedSearch.last_used.desc().nullslast(), SavedSearch.created_at.desc()).all()

        results = []
        for search in saved_searches:
            try:
                search_dict = {
                    "id": str(search.id),
                    "name": search.name,
                    "description": search.description,
                    "query_text": search.query_text,
                    "search_type": search.search_type,
                    "filters": json.loads(search.search_filters) if search.search_filters else {},
                    "max_results": search.max_results,
                    "similarity_threshold": search.similarity_threshold,
                    "usage_count": search.usage_count,
                    "created_at": search.created_at.isoformat() if search.created_at else None,
                    "last_used": search.last_used.isoformat() if search.last_used else None,
                    "tags": search.get_tag_list(),
                    "is_public": search.is_public
                }
                results.append(search_dict)
            except Exception as e:
                logger.warning(f"Failed to serialize saved search {search.id}: {e}")
                continue

        logger.debug(f"Retrieved {len(results)} saved searches for user {user_id}")
        return results

    except Exception as e:
        logger.error(f"Failed to get saved searches for user {user_id}: {str(e)}", exc_info=True)
        return []


async def get_recent_searches(user_id: int, limit: int = 10):
    """Get user's recent search history."""
    try:
        from database.connection import get_db

        db = next(get_db())

        recent_searches = db.query(SearchQuery).filter(
            SearchQuery.user_id == user_id
        ).order_by(SearchQuery.created_at.desc()).limit(limit).all()

        results = []
        for search in recent_searches:
            try:
                search_dict = {
                    "id": str(search.id),
                    "query_text": search.query_text,
                    "query_type": search.query_type,
                    "created_at": search.created_at.isoformat() if search.created_at else None,
                    "results_count": search.results_count,
                    "search_time_ms": search.search_time_ms,
                    "filters": json.loads(search.search_filters) if search.search_filters else {}
                }
                results.append(search_dict)
            except Exception as e:
                logger.warning(f"Failed to serialize recent search {search.id}: {e}")
                continue

        logger.debug(f"Retrieved {len(results)} recent searches for user {user_id}")
        return results

    except Exception as e:
        logger.error(f"Failed to get recent searches for user {user_id}: {str(e)}", exc_info=True)
        return []