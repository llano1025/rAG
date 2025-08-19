"""
Hugging Face cross-encoder reranker implementation.

This module provides reranking capabilities using Hugging Face cross-encoder models
for improved search result relevance scoring.
"""

import logging
import asyncio
import torch
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
import numpy as np

try:
    from sentence_transformers import CrossEncoder
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    CrossEncoder = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None

from .base_reranker import BaseReranker, RerankResult, SearchResult

logger = logging.getLogger(__name__)


class HuggingFaceReranker(BaseReranker):
    """
    Hugging Face cross-encoder reranker for search result re-scoring.
    
    Supports popular models like:
    - ms-marco-MiniLM-L-6-v2
    - bge-reranker-base
    - bge-reranker-large
    - jina-reranker-v1-base-en
    """
    
    # Popular reranker models with their properties
    POPULAR_MODELS = {
        'ms-marco-MiniLM-L-6-v2': {
            'full_name': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
            'description': 'Efficient MS MARCO trained cross-encoder',
            'max_length': 512,
            'performance_tier': 'fast'
        },
        'bge-reranker-base': {
            'full_name': 'BAAI/bge-reranker-base',
            'description': 'BGE base reranker model',
            'max_length': 512,
            'performance_tier': 'balanced'
        },
        'bge-reranker-large': {
            'full_name': 'BAAI/bge-reranker-large',
            'description': 'BGE large reranker model (highest quality)',
            'max_length': 512,
            'performance_tier': 'accurate'
        },
        'jina-reranker-v1-base-en': {
            'full_name': 'jinaai/jina-reranker-v1-base-en',
            'description': 'Jina AI English reranker',
            'max_length': 512,
            'performance_tier': 'balanced'
        }
    }
    
    def __init__(
        self, 
        model_name: str,
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 32,
        use_sentence_transformers: bool = True,
        **config
    ):
        """
        Initialize Hugging Face reranker.
        
        Args:
            model_name: Name or path of the reranker model
            device: Device to run model on ('cuda', 'cpu', or None for auto)
            max_length: Maximum sequence length
            batch_size: Batch size for processing
            use_sentence_transformers: Use sentence-transformers CrossEncoder
            **config: Additional configuration
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers and transformers packages are required")
        
        super().__init__(model_name, **config)
        
        # Resolve model name if it's a popular model alias
        self.full_model_name = self._resolve_model_name(model_name)
        
        # Device setup
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.max_length = max_length
        self.batch_size = batch_size
        self.use_sentence_transformers = use_sentence_transformers
        
        # Model components
        self.cross_encoder = None
        self.tokenizer = None
        self.model = None
        
        logger.info(f"Initialized HuggingFace reranker: {self.full_model_name} on {self.device}")
    
    def _resolve_model_name(self, model_name: str) -> str:
        """Resolve model name from alias to full name."""
        if model_name in self.POPULAR_MODELS:
            return self.POPULAR_MODELS[model_name]['full_name']
        return model_name
    
    async def load_model(self) -> bool:
        """Load the cross-encoder model."""
        try:
            if self.is_loaded:
                logger.info(f"Model {self.full_model_name} already loaded")
                return True
            
            logger.info(f"Loading reranker model: {self.full_model_name}")
            
            if self.use_sentence_transformers:
                # Use sentence-transformers CrossEncoder (recommended)
                self.cross_encoder = CrossEncoder(
                    self.full_model_name,
                    max_length=self.max_length,
                    device=self.device
                )
            else:
                # Use transformers directly
                self.tokenizer = AutoTokenizer.from_pretrained(self.full_model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.full_model_name)
                self.model.to(self.device)
                self.model.eval()
            
            self.is_loaded = True
            logger.info(f"Successfully loaded reranker model: {self.full_model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load reranker model {self.full_model_name}: {e}")
            self.is_loaded = False
            return False
    
    async def unload_model(self) -> bool:
        """Unload the model to free memory."""
        try:
            if not self.is_loaded:
                return True
                
            if self.cross_encoder:
                del self.cross_encoder
                self.cross_encoder = None
                
            if self.model:
                del self.model
                self.model = None
                
            if self.tokenizer:
                del self.tokenizer
                self.tokenizer = None
            
            # Clear GPU cache if using CUDA
            if self.device.startswith('cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.is_loaded = False
            logger.info(f"Unloaded reranker model: {self.full_model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unload reranker model: {e}")
            return False
    
    async def rerank(
        self,
        query: str,
        results: List[Union[SearchResult, Dict[str, Any]]],
        top_k: Optional[int] = None,
        score_weight: float = 0.5,
        min_rerank_score: Optional[float] = None
    ) -> List[RerankResult]:
        """
        Rerank search results using cross-encoder model.
        
        Args:
            query: Search query
            results: List of search results to rerank
            top_k: Maximum number of results to return
            score_weight: Weight for rerank scores in final score
            min_rerank_score: Minimum rerank score threshold
            
        Returns:
            List of reranked results
        """
        if not self.is_loaded:
            logger.warning("Model not loaded, loading now...")
            if not await self.load_model():
                raise RuntimeError(f"Failed to load reranker model: {self.full_model_name}")
        
        if not results:
            return []
        
        start_time = datetime.now()
        
        # Normalize input results
        search_results = [self._normalize_search_result(r) for r in results]
        
        # Extract texts for reranking
        texts = [result.text for result in search_results]
        
        # Get rerank scores
        rerank_scores = await self.get_relevance_scores(query, texts)
        
        # Create reranked results
        reranked_results = []
        rerank_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        for i, (result, rerank_score) in enumerate(zip(search_results, rerank_scores)):
            # Skip results below threshold
            if min_rerank_score is not None and rerank_score < min_rerank_score:
                continue
            
            # Calculate combined score
            combined_score = self._combine_scores(
                result.score,
                rerank_score,
                score_weight
            )
            
            # Create rerank result
            rerank_result = RerankResult(
                document_id=result.document_id,
                chunk_id=result.chunk_id,
                text=result.text,
                original_score=result.score,
                rerank_score=rerank_score,
                combined_score=combined_score,
                metadata={
                    **result.metadata,
                    'reranker_model': self.full_model_name,
                    'original_position': i
                },
                document_metadata=result.document_metadata,
                highlight=result.highlight,
                reranker_model=self.full_model_name,
                rerank_time_ms=rerank_time_ms
            )
            
            reranked_results.append(rerank_result)
        
        # Sort by combined score (descending)
        reranked_results.sort(key=lambda x: x.combined_score, reverse=True)
        
        # Apply top_k limit
        if top_k:
            reranked_results = reranked_results[:top_k]
        
        logger.info(
            f"Reranked {len(results)} results to {len(reranked_results)} results "
            f"in {rerank_time_ms:.2f}ms using {self.full_model_name}"
        )
        
        return reranked_results
    
    async def batch_rerank(
        self,
        queries: List[str],
        results_list: List[List[Union[SearchResult, Dict[str, Any]]]],
        top_k: Optional[int] = None,
        score_weight: float = 0.5,
        min_rerank_score: Optional[float] = None
    ) -> List[List[RerankResult]]:
        """Batch rerank multiple result sets."""
        if not self.is_loaded:
            if not await self.load_model():
                raise RuntimeError(f"Failed to load reranker model: {self.full_model_name}")
        
        # Process each query-results pair
        batch_results = []
        for query, results in zip(queries, results_list):
            reranked = await self.rerank(
                query, results, top_k, score_weight, min_rerank_score
            )
            batch_results.append(reranked)
        
        return batch_results
    
    async def get_relevance_scores(
        self,
        query: str,
        documents: List[str]
    ) -> List[float]:
        """Get relevance scores for query-document pairs."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        if not documents:
            return []
        
        try:
            # Prepare query-document pairs
            pairs = [(query, doc) for doc in documents]
            
            if self.use_sentence_transformers and self.cross_encoder:
                # Use sentence-transformers CrossEncoder
                scores = self.cross_encoder.predict(pairs, batch_size=self.batch_size)
                
                # Convert to list and ensure proper float type
                if isinstance(scores, np.ndarray):
                    scores = scores.tolist()
                
                # Normalize scores to 0-1 range using sigmoid
                normalized_scores = [self._sigmoid(score) for score in scores]
                
            else:
                # Use transformers directly
                normalized_scores = await self._predict_with_transformers(pairs)
            
            return normalized_scores
            
        except Exception as e:
            logger.error(f"Failed to get relevance scores: {e}")
            # Return default scores to prevent complete failure
            return [0.5] * len(documents)
    
    async def _predict_with_transformers(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Predict scores using transformers directly."""
        scores = []
        
        # Process in batches
        for i in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[i:i + self.batch_size]
            
            # Tokenize batch
            batch_queries = [pair[0] for pair in batch_pairs]
            batch_docs = [pair[1] for pair in batch_pairs]
            
            inputs = self.tokenizer(
                batch_queries,
                batch_docs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Convert to probabilities and take positive class
                if logits.shape[1] == 1:
                    # Single output score
                    batch_scores = torch.sigmoid(logits.squeeze()).cpu().numpy()
                else:
                    # Multi-class output, take positive class probability
                    probs = torch.softmax(logits, dim=-1)
                    batch_scores = probs[:, 1].cpu().numpy()
                
                scores.extend(batch_scores.tolist())
        
        return scores
    
    def _sigmoid(self, x: float) -> float:
        """Apply sigmoid normalization to score."""
        try:
            return 1 / (1 + np.exp(-x))
        except (OverflowError, FloatingPointError):
            return 0.0 if x < 0 else 1.0
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information."""
        base_info = super().get_model_info()
        
        # Add HuggingFace specific info
        model_config = self.POPULAR_MODELS.get(self.model_name, {})
        
        hf_info = {
            'full_model_name': self.full_model_name,
            'device': self.device,
            'max_length': self.max_length,
            'batch_size': self.batch_size,
            'use_sentence_transformers': self.use_sentence_transformers,
            'performance_tier': model_config.get('performance_tier', 'unknown'),
            'description': model_config.get('description', 'Custom reranker model')
        }
        
        base_info.update(hf_info)
        return base_info
    
    @classmethod
    def get_available_models(cls) -> Dict[str, Dict[str, Any]]:
        """Get list of available popular models."""
        return cls.POPULAR_MODELS.copy()
    
    @classmethod
    def create_from_model_name(
        cls,
        model_name: str,
        device: Optional[str] = None,
        **kwargs
    ) -> 'HuggingFaceReranker':
        """
        Create reranker instance from model name.
        
        Args:
            model_name: Model name or alias
            device: Device to use
            **kwargs: Additional configuration
            
        Returns:
            HuggingFaceReranker instance
        """
        # Get model-specific configuration
        model_config = cls.POPULAR_MODELS.get(model_name, {})
        
        # Merge with provided kwargs
        config = {
            'max_length': model_config.get('max_length', 512),
            **kwargs
        }
        
        return cls(model_name, device=device, **config)