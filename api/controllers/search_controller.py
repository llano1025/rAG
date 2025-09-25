from typing import List, Optional, Dict
from sqlalchemy.ext.asyncio import AsyncSession
import json
import logging
from datetime import datetime, timezone

from vector_db.embedding_manager import EmbeddingManager
from vector_db.search_optimizer import SearchOptimizer
from vector_db.search_context_processor import ContextProcessor
from vector_db.search_manager import EnhancedSearchEngine
from api.schemas.search_schemas import SearchType, SearchFilters
from database.models import User, SearchQuery, Document, SavedSearch
from utils.security.audit_logger import AuditLogger
from database.connection import get_db

logger = logging.getLogger(__name__)


class SearchController:
    """Controller for all search operations."""

    def __init__(
        self,
        search_engine: EnhancedSearchEngine,
        db: AsyncSession,
        audit_logger: AuditLogger,
        # Keep legacy components for backward compatibility
        embedding_manager: Optional[EmbeddingManager] = None,
        content_search: Optional[SearchOptimizer] = None,
        context_search: Optional[SearchOptimizer] = None,
        context_processor: Optional[ContextProcessor] = None
    ):
        self.search_engine = search_engine
        self.db = db
        self.audit_logger = audit_logger
        # Legacy components (kept for backward compatibility)
        self.embedding_manager = embedding_manager
        self.content_search = content_search
        self.context_search = context_search
        self.context_processor = context_processor

    async def search(
        self,
        query: str,
        user: User,
        search_type: str = "contextual",
        filters: Optional[SearchFilters] = None,
        top_k: int = 20,
        similarity_threshold: Optional[float] = None,
        embedding_model: Optional[str] = None,
    ) -> List[Dict]:
        """Unified search endpoint supporting all search types with MMR and reranker."""
        start_time = datetime.now(timezone.utc)

        try:
            # Use filters directly (already converted by route)
            search_filters = filters or SearchFilters()

            # Set search parameters
            if similarity_threshold is not None:
                search_filters.min_score = similarity_threshold

            # Set embedding model if provided
            if embedding_model is not None:
                search_filters.embedding_model = embedding_model

            # Execute search based on type
            if search_type == "semantic":
                results = await self.search_engine.search(
                    query=query,
                    user=user,
                    filters=search_filters,
                    search_type=SearchType.SEMANTIC,
                    limit=top_k,
                    db=self.db
                )
            elif search_type == "contextual":
                results = await self.search_engine.search(
                    query=query,
                    user=user,
                    filters=search_filters,
                    search_type=SearchType.CONTEXTUAL,
                    limit=top_k,
                    db=self.db
                )
            elif search_type == "text":
                results = await self.search_engine.search(
                    query=query,
                    user=user,
                    filters=search_filters,
                    search_type=SearchType.KEYWORD,
                    limit=top_k,
                    db=self.db
                )
            else:
                # Default to contextual search
                results = await self.search_engine.search(
                    query=query,
                    user=user,
                    filters=search_filters,
                    search_type=SearchType.CONTEXTUAL,
                    limit=top_k,
                    db=self.db
                )
            return results

        except Exception as e:
            logger.error(f"Failed to perform search by controller: {e}")
            return []  # Return empty list instead of None


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
                logger.debug(f"Failed to serialize recent search {search.id}: {e}")
                continue

        logger.debug(f"Retrieved {len(results)} recent searches for user {user_id}")
        return results

    except Exception as e:
        logger.error(f"Failed to get recent searches for user {user_id}: {str(e)}", exc_info=True)
        return []