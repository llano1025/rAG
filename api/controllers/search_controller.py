# # api/controllers/search_controller.py
# from typing import List, Optional
# from fastapi import HTTPException

# from vector_db.search_optimizer import SearchOptimizer
# from llm.response_handler import ResponseHandler

# class SearchController:
#     def __init__(
#         self,
#         search_optimizer: SearchOptimizer,
#         response_handler: ResponseHandler
#     ):
#         self.search_optimizer = search_optimizer
#         self.response_handler = response_handler

#     async def search(
#         self,
#         query: str,
#         filters: Optional[dict] = None,
#         limit: int = 10
#     ) -> List[dict]:
#         try:
#             results = await self.search_optimizer.search(
#                 query=query,
#                 filters=filters,
#                 limit=limit
#             )
#             return await self.response_handler.format_results(results)
#         except Exception as e:
#             raise HTTPException(status_code=500, detail=str(e))
# /controllers/search_controller.py

# /controllers/search_controller.py
from typing import List, Dict, Optional, Tuple
from fastapi import HTTPException
import numpy as np

from vector_db.embedding_manager import EmbeddingManager
from vector_db.search_optimizer import SearchOptimizer, SearchError
from vector_db.context_processor import ContextProcessor
from vector_db.chunking import Chunk

class SearchController:
    """Controller for all search operations."""
    
    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        content_search: SearchOptimizer,
        context_search: SearchOptimizer,
        context_processor: ContextProcessor
    ):
        self.embedding_manager = embedding_manager
        self.content_search = content_search
        self.context_search = context_search
        self.context_processor = context_processor

    async def search(
        self,
        query: str,
        query_context: Optional[str] = None,
        k: int = 5,
        filters: Optional[Dict] = None,
        content_weight: float = 0.7,
        context_weight: float = 0.3,
        min_score: float = 0.0
    ) -> List[Dict]:
        """Unified search endpoint with all features."""
        try:
            # Process query context
            processed_context = self.context_processor.process_context(
                query, query_context
            )
            
            # Generate embeddings
            query_emb, context_emb = await self.embedding_manager.generate_query_embeddings(
                query, processed_context
            )
            
            # Search in both spaces
            content_results = self.content_search.search(
                query_emb, k=k, min_score=min_score
            )
            context_results = self.context_search.search(
                context_emb, k=k, min_score=min_score
            )
            
            # Combine and filter results
            results = self._combine_and_filter_results(
                content_results,
                context_results,
                content_weight,
                context_weight,
                filters
            )
            
            return self._format_results(results[:k])
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Search failed: {str(e)}"
            )

    async def batch_search(
        self,
        queries: List[str],
        contexts: Optional[List[str]] = None,
        k: int = 5,
        filters: Optional[Dict] = None,
        content_weight: float = 0.7,
        context_weight: float = 0.3,
        min_score: float = 0.0
    ) -> List[List[Dict]]:
        """Unified batch search endpoint."""
        try:
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
            for q_content, q_context in zip(content_results, context_results):
                combined = self._combine_and_filter_results(
                    q_content,
                    q_context,
                    content_weight,
                    context_weight,
                    filters
                )
                batch_results.append(self._format_results(combined[:k]))
            
            return batch_results
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Batch search failed: {str(e)}"
            )

    def _combine_and_filter_results(
        self,
        content_results: List[Tuple[Chunk, float]],
        context_results: List[Tuple[Chunk, float]],
        content_weight: float,
        context_weight: float,
        filters: Optional[Dict]
    ) -> List[Tuple[Chunk, float]]:
        """Combine, filter, and rank search results."""
        combined_scores = {}
        
        for results, weight in [
            (content_results, content_weight),
            (context_results, context_weight)
        ]:
            for chunk, score in results:
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