# # api/controllers/vector_controller.py
# from typing import List, Dict, Optional
# from fastapi import HTTPException
# import numpy as np

# from vector_db.chunking import AdaptiveChunker, Chunk, ChunkingError
# from vector_db.embedding_manager import EmbeddingManager
# from vector_db.search_optimizer import SearchOptimizer, SearchError
# from vector_db.version_controller import VersionController

# class VectorController:
#     def __init__(
#         self,
#         chunker: AdaptiveChunker,
#         embedding_manager: EmbeddingManager,
#         search_optimizer: SearchOptimizer,
#         version_controller: VersionController
#     ):
#         self.chunker = chunker
#         self.embedding_manager = embedding_manager
#         self.search_optimizer = search_optimizer
#         self.version_controller = version_controller

#     async def process_document(
#         self,
#         content: str,
#         document_id: str,
#         metadata: Dict
#     ) -> dict:
#         try:
#             # Create chunks using adaptive chunking
#             chunks = self.chunker.chunk_document(content, metadata)
            
#             # Generate embeddings for all chunks at once
#             chunk_texts = [chunk.text for chunk in chunks]
#             embeddings = await self.embedding_manager.generate_embeddings(chunk_texts)
            
#             # Create new chunks with embeddings
#             chunks_with_embeddings = [
#                 Chunk(
#                     text=chunk.text,
#                     start_idx=chunk.start_idx,
#                     end_idx=chunk.end_idx,
#                     metadata=chunk.metadata,
#                     embedding=np.array(embedding)
#                 )
#                 for chunk, embedding in zip(chunks, embeddings)
#             ]
            
#             # Build search index with new chunks
#             self.search_optimizer.build_index(chunks_with_embeddings)
            
#             # Version control
#             version_info = await self.version_controller.create_version(
#                 document_id,
#                 chunks_with_embeddings
#             )
            
#             return {
#                 "document_id": document_id,
#                 "chunk_count": len(chunks_with_embeddings),
#                 "version": version_info,
#                 "chunks": [
#                     {
#                         "text": chunk.text,
#                         "start_idx": chunk.start_idx,
#                         "end_idx": chunk.end_idx,
#                         "metadata": chunk.metadata
#                     }
#                     for chunk in chunks_with_embeddings
#                 ]
#             }
#         except ChunkingError as e:
#             raise HTTPException(status_code=500, detail=f"Chunking failed: {str(e)}")
#         except SearchError as e:
#             raise HTTPException(status_code=500, detail=f"Search index building failed: {str(e)}")
#         except Exception as e:
#             raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

#     async def search_similar(
#         self,
#         query: str,
#         k: int = 5
#     ) -> List[dict]:
#         try:
#             # Generate embedding for query
#             query_embeddings = await self.embedding_manager.generate_embeddings([query])
#             query_embedding = np.array(query_embeddings[0])
            
#             # Search similar chunks
#             results = self.search_optimizer.search(query_embedding, k=k)
            
#             # Format results
#             formatted_results = [
#                 {
#                     "text": chunk.text,
#                     "start_idx": chunk.start_idx,
#                     "end_idx": chunk.end_idx,
#                     "metadata": chunk.metadata,
#                     "similarity_score": float(score)
#                 }
#                 for chunk, score in results
#             ]
            
#             return formatted_results
#         except SearchError as e:
#             raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
#         except Exception as e:
#             raise HTTPException(status_code=500, detail=f"Search operation failed: {str(e)}")

# /controllers/vector_controller.py
from typing import Dict, List
from fastapi import HTTPException

from vector_db.chunking import AdaptiveChunker, Chunk, ChunkingError
from vector_db.embedding_manager import EmbeddingManager
from vector_db.search_optimizer import SearchOptimizer, SearchConfig, MetricType, IndexType
from vector_db.version_controller import VersionController

class VectorController:
    """Controller for managing document vectors and indexing."""
    
    def __init__(
        self,
        chunker: AdaptiveChunker,
        embedding_manager: EmbeddingManager,
        version_controller: VersionController,
        embedding_dim: int = 768
    ):
        self.chunker = chunker
        self.embedding_manager = embedding_manager
        self.version_controller = version_controller
        
        # Create content and context search optimizers
        self.content_search = self._create_search_optimizer(embedding_dim)
        self.context_search = self._create_search_optimizer(embedding_dim)

    def _create_search_optimizer(self, embedding_dim: int) -> SearchOptimizer:
        """Create a search optimizer with standard configuration."""
        config = SearchConfig(
            dimension=embedding_dim,
            index_type=IndexType.HNSW,
            metric=MetricType.COSINE,
            ef_search=64,
            ef_construction=200
        )
        return SearchOptimizer(config)

    async def process_document(
        self,
        content: str,
        document_id: str,
        metadata: Dict
    ) -> dict:
        """Process and index document with contextual awareness."""
        try:
            # Create chunks with context
            chunks = self.chunker.chunk_document(content, metadata)
            
            # Generate content and context embeddings
            chunks_with_embeddings = await self.embedding_manager.generate_embeddings(chunks)
            
            # Build search indices
            self.content_search.build_index(chunks_with_embeddings)
            self.context_search.build_index(chunks_with_embeddings)
            
            # Version control
            version_info = await self.version_controller.create_version(
                document_id,
                chunks_with_embeddings,
                metadata
            )
            
            return {
                "document_id": document_id,
                "chunk_count": len(chunks_with_embeddings),
                "version": version_info,
                "chunks": [
                    {
                        "text": chunk.text,
                        "context": chunk.context_text,
                        "start_idx": chunk.start_idx,
                        "end_idx": chunk.end_idx,
                        "metadata": chunk.metadata
                    }
                    for chunk in chunks_with_embeddings
                ]
            }
            
        except ChunkingError as e:
            raise HTTPException(status_code=500, detail=f"Chunking failed: {str(e)}")
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process document: {str(e)}"
            )

    def get_search_optimizers(self) -> tuple[SearchOptimizer, SearchOptimizer]:
        """Get content and context search optimizers for search operations."""
        return self.content_search, self.context_search