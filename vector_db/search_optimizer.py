from typing import List, Tuple, Optional, Dict
import logging
from dataclasses import dataclass
from enum import Enum
import time

try:
    import numpy as np
    import faiss
    SEARCH_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Search optimization dependencies not available: {e}")
    np = None
    faiss = None
    SEARCH_AVAILABLE = False

from .chunking import Chunk

class MetricType(Enum):
    COSINE = "cosine"
    L2 = "l2"
    INNER_PRODUCT = "inner_product"

class IndexType(Enum):
    FLAT = "flat"
    IVF = "ivf"
    HNSW = "hnsw"
    IVF_PQ = "ivf_pq"

@dataclass
class SearchConfig:
    dimension: int
    index_type: IndexType
    metric: MetricType
    n_clusters: int = 100
    n_probes: int = 8  # Number of clusters to search
    ef_search: int = 64  # HNSW search depth
    ef_construction: int = 200  # HNSW construction depth
    pq_m: int = 8  # Number of PQ sub-vectors
    nlist: Optional[int] = None  # Number of IVF clusters, computed automatically if None

class SearchError(Exception):
    """Raised when search operation fails."""
    pass

class SearchOptimizer:
    """Enhanced vector similarity search with multiple index types and optimizations."""
    
    def __init__(self, config: SearchConfig):
        if not SEARCH_AVAILABLE:
            raise ImportError("numpy and faiss are required for search optimization functionality")
        
        self.config = config
        self.index = None
        self.chunks = []
        self.id_to_chunk: Dict[int, Chunk] = {}
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
    def _setup_logging(self):
        """Configure logging with performance metrics."""
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors for cosine similarity."""
        return vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]

    def _create_index(self) -> faiss.Index:
        """Create an optimized FAISS index based on configuration."""
        d = self.config.dimension
        
        if self.config.metric == MetricType.COSINE:
            metric = faiss.METRIC_INNER_PRODUCT
        else:
            metric = faiss.METRIC_L2

        if self.config.index_type == IndexType.FLAT:
            return faiss.IndexFlatIP(d) if metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(d)
            
        elif self.config.index_type == IndexType.HNSW:
            index = faiss.IndexHNSWFlat(d, self.config.ef_construction, metric)
            index.hnsw.efSearch = self.config.ef_search
            return index
            
        elif self.config.index_type == IndexType.IVF:
            # Compute optimal number of clusters if not specified
            if not self.config.nlist:
                self.config.nlist = int(np.sqrt(len(self.chunks)))
            
            quantizer = faiss.IndexFlatL2(d)
            index = faiss.IndexIVFFlat(quantizer, d, self.config.nlist, metric)
            index.nprobe = self.config.n_probes
            return index
            
        elif self.config.index_type == IndexType.IVF_PQ:
            quantizer = faiss.IndexFlatL2(d)
            index = faiss.IndexIVFPQ(
                quantizer, d, self.config.nlist or int(np.sqrt(len(self.chunks))),
                self.config.pq_m, 8  # 8 bits per component
            )
            index.nprobe = self.config.n_probes
            return index
            
        raise ValueError(f"Unsupported index type: {self.config.index_type}")

    def build_index(self, chunks: List[Chunk]):
        """Build optimized search index from chunks."""
        try:
            start_time = time.time()
            self.chunks = chunks
            
            # Create ID mapping
            self.id_to_chunk = {i: chunk for i, chunk in enumerate(chunks)}
            
            # Stack embeddings
            embeddings = np.vstack([chunk.embedding for chunk in chunks])
            
            # Normalize if using cosine similarity
            if self.config.metric == MetricType.COSINE:
                embeddings = self._normalize_vectors(embeddings)
            
            # Create and train index
            self.index = self._create_index()
            
            if not self.index.is_trained:
                self.logger.info("Training index...")
                self.index.train(embeddings)
            
            self.logger.info("Adding vectors to index...")
            self.index.add(embeddings)
            
            build_time = time.time() - start_time
            self.logger.info(f"Index built in {build_time:.2f} seconds")
            
            # Log index statistics
            self._log_index_stats()
            
        except Exception as e:
            self.logger.error(f"Failed to build search index: {str(e)}")
            raise SearchError(f"Index building failed: {str(e)}")

    def _log_index_stats(self):
        """Log important index statistics."""
        self.logger.info(f"Index type: {self.config.index_type}")
        self.logger.info(f"Number of vectors: {len(self.chunks)}")
        self.logger.info(f"Vector dimension: {self.config.dimension}")
        
        if hasattr(self.index, 'nprobe'):
            self.logger.info(f"Number of probes: {self.index.nprobe}")
        if hasattr(self.index, 'hnsw'):
            self.logger.info(f"HNSW ef_search: {self.index.hnsw.efSearch}")

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        min_score: float = -1.0
    ) -> List[Tuple[Chunk, float]]:
        """
        Enhanced search for most similar chunks to query embedding.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            min_score: Minimum similarity score threshold
            
        Returns:
            List of (chunk, score) tuples sorted by similarity
        """
        try:
            start_time = time.time()
            
            # Normalize query if using cosine similarity
            if self.config.metric == MetricType.COSINE:
                query_embedding = self._normalize_vectors(
                    query_embedding.reshape(1, -1)
                )
            
            # Perform search
            distances, indices = self.index.search(
                query_embedding.reshape(1, -1), k
            )
            
            # Process results
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx != -1:  # Skip invalid results
                    # Convert distance to similarity score if needed
                    score = (
                        distance if self.config.metric == MetricType.COSINE
                        else 1 / (1 + distance)  # Convert L2 distance to similarity
                    )
                    
                    if score >= min_score:
                        results.append((self.id_to_chunk[idx], float(score)))
            
            # Sort by score in descending order
            results.sort(key=lambda x: x[1], reverse=True)
            
            search_time = time.time() - start_time
            self.logger.debug(
                f"Search completed in {search_time:.4f} seconds, "
                f"found {len(results)} results"
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Search operation failed: {str(e)}")
            raise SearchError(f"Search failed: {str(e)}")

    def batch_search(
        self,
        query_embeddings: np.ndarray,
        k: int = 5,
        min_score: float = -1.0
    ) -> List[List[Tuple[Chunk, float]]]:
        """
        Perform batch search for multiple query vectors.
        
        Args:
            query_embeddings: Matrix of query vectors
            k: Number of results per query
            min_score: Minimum similarity score threshold
            
        Returns:
            List of search results for each query
        """
        try:
            start_time = time.time()
            
            # Normalize queries if using cosine similarity
            if self.config.metric == MetricType.COSINE:
                query_embeddings = self._normalize_vectors(query_embeddings)
            
            # Perform batch search
            distances, indices = self.index.search(query_embeddings, k)
            
            # Process results for each query
            batch_results = []
            for query_distances, query_indices in zip(distances, indices):
                results = []
                for idx, distance in zip(query_indices, query_distances):
                    if idx != -1:
                        score = (
                            distance if self.config.metric == MetricType.COSINE
                            else 1 / (1 + distance)
                        )
                        if score >= min_score:
                            results.append((self.id_to_chunk[idx], float(score)))
                results.sort(key=lambda x: x[1], reverse=True)
                batch_results.append(results)
            
            batch_time = time.time() - start_time
            self.logger.debug(
                f"Batch search completed in {batch_time:.4f} seconds, "
                f"processed {len(query_embeddings)} queries"
            )
            
            return batch_results
            
        except Exception as e:
            self.logger.error(f"Batch search operation failed: {str(e)}")
            raise SearchError(f"Batch search failed: {str(e)}")

    def contextual_search(
        self,
        query_embedding: np.ndarray,
        context_embedding: np.ndarray,
        k: int = 5,
        filters: Optional[Dict] = None,
        content_weight: float = 0.7,
        context_weight: float = 0.3
    ) -> List[Tuple[Chunk, float]]:
        """
        Perform contextual search using both content and context embeddings.
        """
        try:
            # Search in content space
            content_results = self.search(
                query_embedding,
                k=k * 2,  # Get more results for reranking
                min_score=-1.0
            )
            
            # Search in context space
            context_results = self.search(
                context_embedding,
                k=k * 2,
                min_score=-1.0
            )
            
            # Combine results with weights
            combined_scores = {}
            
            # Process content results
            for chunk, score in content_results:
                if self._apply_filters(chunk, filters):
                    chunk_id = f"{chunk.start_idx}_{chunk.end_idx}"
                    combined_scores[chunk_id] = {
                        "chunk": chunk,
                        "score": score * content_weight
                    }
            
            # Process context results
            for chunk, score in context_results:
                if self._apply_filters(chunk, filters):
                    chunk_id = f"{chunk.start_idx}_{chunk.end_idx}"
                    if chunk_id in combined_scores:
                        combined_scores[chunk_id]["score"] += score * context_weight
                    else:
                        combined_scores[chunk_id] = {
                            "chunk": chunk,
                            "score": score * context_weight
                        }
            
            # Convert to list and sort
            results = [
                (item["chunk"], item["score"])
                for item in combined_scores.values()
            ]
            results.sort(key=lambda x: x[1], reverse=True)
            
            return results[:k]
            
        except Exception as e:
            self.logger.error(f"Contextual search failed: {str(e)}")
            raise SearchError(f"Contextual search failed: {str(e)}")

    def batch_contextual_search(
        self,
        query_embeddings: np.ndarray,
        context_embeddings: np.ndarray,
        k: int = 5,
        filters: Optional[Dict] = None,
        content_weight: float = 0.7,
        context_weight: float = 0.3
    ) -> List[List[Tuple[Chunk, float]]]:
        """
        Perform batch contextual search for multiple queries.
        """
        try:
            # Batch search in content space
            content_results = self.batch_search(
                query_embeddings,
                k=k * 2
            )
            
            # Batch search in context space
            context_results = self.batch_search(
                context_embeddings,
                k=k * 2
            )
            
            # Combine results for each query
            batch_results = []
            for i in range(len(query_embeddings)):
                results = self._combine_results(
                    content_results[i],
                    context_results[i],
                    content_weight,
                    context_weight,
                    filters
                )
                batch_results.append(results[:k])
            
            return batch_results
            
        except Exception as e:
            self.logger.error(f"Batch contextual search failed: {str(e)}")
            raise SearchError(f"Batch contextual search failed: {str(e)}")

    def _combine_results(
        self,
        content_results: List[Tuple[Chunk, float]],
        context_results: List[Tuple[Chunk, float]],
        content_weight: float,
        context_weight: float,
        filters: Optional[Dict]
    ) -> List[Tuple[Chunk, float]]:
        """Combine and rerank results from content and context search."""
        combined_scores = {}
        
        # Process content results
        for chunk, score in content_results:
            if self._apply_filters(chunk, filters):
                chunk_id = f"{chunk.start_idx}_{chunk.end_idx}"
                combined_scores[chunk_id] = {
                    "chunk": chunk,
                    "score": score * content_weight
                }
        
        # Process context results
        for chunk, score in context_results:
            if self._apply_filters(chunk, filters):
                chunk_id = f"{chunk.start_idx}_{chunk.end_idx}"
                if chunk_id in combined_scores:
                    combined_scores[chunk_id]["score"] += score * context_weight
                else:
                    combined_scores[chunk_id] = {
                        "chunk": chunk,
                        "score": score * context_weight
                    }
        
        # Convert to list and sort
        results = [
            (item["chunk"], item["score"])
            for item in combined_scores.values()
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results

    def _apply_filters(self, chunk: Chunk, filters: Optional[Dict]) -> bool:
        """Apply metadata filters to chunk."""
        if not filters:
            return True
            
        for key, value in filters.items():
            if key not in chunk.metadata:
                return False
            if isinstance(value, list):
                if chunk.metadata[key] not in value:
                    return False
            elif chunk.metadata[key] != value:
                return False
        
        return True