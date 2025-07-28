# vector_db/embedding_manager.py

from typing import List, Dict, Any, Optional, Union, Tuple
import logging
import asyncio
import aiohttp
from datetime import datetime
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy.orm import Session

try:
    import torch
    from transformers import AutoModel, AutoTokenizer
    import numpy as np
    from sentence_transformers import SentenceTransformer
    TORCH_AVAILABLE = True
except ImportError as e:
    logging.warning(f"PyTorch not available: {e}. Embedding functionality will be limited.")
    TORCH_AVAILABLE = False

# Legacy EmbeddingManager and EmbeddingError for backward compatibility
class EmbeddingError(Exception):
    """Raised when embedding generation fails."""
    pass

# For backward compatibility, alias EnhancedEmbeddingManager as EmbeddingManager at module level
EmbeddingManager = None  # Will be set after class definition

from .chunking import Chunk
from database.models import Document, DocumentChunk

logger = logging.getLogger(__name__)

class BaseEmbeddingProvider(ABC):
    """Base class for embedding providers."""
    
    @abstractmethod
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this provider."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        pass

class HuggingFaceEmbeddingProvider(BaseEmbeddingProvider):
    """HuggingFace transformers-based embedding provider."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: str = None,
        batch_size: int = 32
    ):
        if not TORCH_AVAILABLE:
            raise EmbeddingError("PyTorch and transformers are required for HuggingFace embeddings. Please install: pip install torch transformers sentence-transformers")
        
        # Validate inputs
        if not model_name or not isinstance(model_name, str):
            raise EmbeddingError("model_name must be a non-empty string")
        
        if batch_size <= 0:
            raise EmbeddingError("batch_size must be positive")
        
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = min(batch_size, 64)  # Cap batch size for stability
        
        try:
            logging.info(f"Initializing HuggingFace model {model_name} on {self.device}")
            
            # Try sentence-transformers first for better performance
            if "sentence-transformers" in model_name:
                self.model = SentenceTransformer(model_name, device=self.device)
                self.model_type = "sentence_transformer"
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                logging.info(f"Successfully loaded sentence-transformer model with {self.embedding_dim} dimensions")
            else:
                # Fallback to raw transformers
                self.model = AutoModel.from_pretrained(model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model.to(self.device)
                self.model.eval()
                self.model_type = "transformer"
                # Estimate embedding dimension (common values)
                self.embedding_dim = self.model.config.hidden_size
                logging.info(f"Successfully loaded transformer model with {self.embedding_dim} dimensions")
                
        except Exception as e:
            error_msg = f"Failed to initialize HuggingFace model {model_name}: {e}"
            logging.error(error_msg)
            raise EmbeddingError(error_msg) from e

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using HuggingFace models."""
        # Validate inputs
        if not texts:
            raise EmbeddingError("No texts provided for embedding generation")
        
        if not all(isinstance(text, str) for text in texts):
            raise EmbeddingError("All texts must be strings")
        
        # Filter out empty texts
        valid_texts = [text.strip() for text in texts if text and text.strip()]
        if not valid_texts:
            raise EmbeddingError("No valid non-empty texts provided")
        
        try:
            if self.model_type == "sentence_transformer":
                # Use sentence-transformers for better performance
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor() as executor:
                    # Create a wrapper function to properly pass keyword arguments
                    def encode_with_params():
                        try:
                            return self.model.encode(
                                valid_texts,
                                batch_size=self.batch_size,
                                normalize_embeddings=True,
                                convert_to_numpy=True
                            )
                        except Exception as encode_error:
                            logger.error(f"Model encoding failed: {encode_error}")
                            raise encode_error
                    
                    embeddings = await loop.run_in_executor(executor, encode_with_params)
                    
                # Validate embeddings output
                if embeddings is None or len(embeddings) == 0:
                    raise EmbeddingError("Model returned empty embeddings")
                
                # Convert to list and validate dimensions
                embeddings_list = embeddings.tolist()
                if len(embeddings_list) != len(valid_texts):
                    logger.warning(f"Embedding count mismatch: {len(embeddings_list)} vs {len(valid_texts)}")
                
                return embeddings_list
            else:
                # Use raw transformers
                all_embeddings = []
                for i in range(0, len(valid_texts), self.batch_size):
                    batch_texts = valid_texts[i:i + self.batch_size]
                    batch_embeddings = await self._encode_batch_transformer(batch_texts)
                    all_embeddings.extend(batch_embeddings)
                return all_embeddings
                
        except EmbeddingError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in embedding generation: {e}")
            raise EmbeddingError(f"HuggingFace embedding generation failed: {str(e)}")

    async def _encode_batch_transformer(self, texts: List[str]) -> List[List[float]]:
        """Encode batch using raw transformers."""
        loop = asyncio.get_event_loop()
        
        def _encode():
            with torch.no_grad():
                inputs = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)
                
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                # Normalize embeddings
                embeddings = embeddings / np.linalg.norm(
                    embeddings, axis=1, keepdims=True
                )
                
                return embeddings.tolist()
        
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(executor, _encode)

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return self.embedding_dim

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "provider": "huggingface",
            "model_name": self.model_name,
            "model_type": self.model_type,
            "embedding_dimension": self.embedding_dim,
            "device": self.device,
            "batch_size": self.batch_size,
            "supports_gpu": torch.cuda.is_available(),
            "gpu_available": self.device == "cuda"
        }

class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    """Ollama-based embedding provider."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "nomic-embed-text",
        batch_size: int = 32
    ):
        # Validate inputs
        if not base_url or not isinstance(base_url, str):
            raise EmbeddingError("base_url must be a non-empty string")
        
        if not model_name or not isinstance(model_name, str):
            raise EmbeddingError("model_name must be a non-empty string")
            
        if batch_size <= 0:
            raise EmbeddingError("batch_size must be positive")
        
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.batch_size = min(batch_size, 32)  # Cap batch size for Ollama
        self.session = None
        self.embedding_dim = None  # Will be determined dynamically
        
        logging.info(f"Initializing Ollama provider with {base_url} and model {model_name}")
        
    async def _get_session(self):
        """Get or create aiohttp session."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Ollama."""
        try:
            session = await self._get_session()
            all_embeddings = []
            
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                
                # Process each text in the batch
                for text in batch_texts:
                    async with session.post(
                        f"{self.base_url}/api/embeddings",
                        json={
                            "model": self.model_name,
                            "prompt": text
                        }
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            embedding = result["embedding"]
                            all_embeddings.append(embedding)
                            
                            # Set embedding dimension on first successful call
                            if self.embedding_dim is None:
                                self.embedding_dim = len(embedding)
                        else:
                            error_text = await response.text()
                            raise EmbeddingError(f"Ollama API error: {response.status} - {error_text}")
                
                # Small delay between batches to avoid overwhelming Ollama
                await asyncio.sleep(0.1)
            
            return all_embeddings
            
        except Exception as e:
            raise EmbeddingError(f"Ollama embedding generation failed: {str(e)}")

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        if self.embedding_dim is None:
            # Return default dimension, will be updated after first call
            return 768
        return self.embedding_dim

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "provider": "ollama",
            "model_name": self.model_name,
            "base_url": self.base_url,
            "embedding_dimension": self.get_embedding_dimension(),
            "batch_size": self.batch_size,
            "supports_gpu": True,  # Depends on Ollama configuration
            "gpu_available": False  # Cannot determine without querying Ollama
        }

    async def __aenter__(self):
        await self._get_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI-based embedding provider."""
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "text-embedding-ada-002",
        batch_size: int = 32
    ):
        try:
            import openai
        except ImportError:
            raise EmbeddingError("openai package required for OpenAI embeddings. Please install: pip install openai")
        
        # Validate inputs
        if not api_key or not isinstance(api_key, str):
            raise EmbeddingError("api_key must be a non-empty string")
            
        if not model_name or not isinstance(model_name, str):
            raise EmbeddingError("model_name must be a non-empty string")
            
        if batch_size <= 0:
            raise EmbeddingError("batch_size must be positive")
        
        try:
            self.client = openai.AsyncOpenAI(api_key=api_key)
            self.model_name = model_name
            self.batch_size = min(batch_size, 100)  # OpenAI has higher batch limits
            
            logging.info(f"Initializing OpenAI provider with model {model_name}")
        except Exception as e:
            raise EmbeddingError(f"Failed to initialize OpenAI client: {e}") from e
        
        # Model dimension mapping
        self.model_dimensions = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072
        }
        
        self.embedding_dim = self.model_dimensions.get(model_name, 1536)

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API."""
        try:
            all_embeddings = []
            
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                
                response = await self.client.embeddings.create(
                    model=self.model_name,
                    input=batch_texts
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # Small delay between batches to respect rate limits
                await asyncio.sleep(0.1)
            
            return all_embeddings
            
        except Exception as e:
            raise EmbeddingError(f"OpenAI embedding generation failed: {str(e)}")

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return self.embedding_dim

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "provider": "openai",
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "batch_size": self.batch_size,
            "supports_gpu": False,  # Cloud-based
            "gpu_available": False  # Cloud-based
        }

class EnhancedEmbeddingManager:
    """Enhanced embedding manager supporting multiple providers."""
    
    def __init__(self, provider: BaseEmbeddingProvider):
        self.provider = provider
        self.model_info = provider.get_model_info()
        
    @classmethod
    def create_huggingface_manager(
        cls,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: str = None,
        batch_size: int = 32
    ):
        """Create manager with HuggingFace provider."""
        try:
            provider = HuggingFaceEmbeddingProvider(model_name, device, batch_size)
            return cls(provider)
        except Exception as e:
            logging.error(f"Failed to create HuggingFace embedding manager: {e}")
            raise EmbeddingError(f"HuggingFace embedding manager creation failed: {e}")from e
    
    @classmethod
    def create_ollama_manager(
        cls,
        base_url: str = "http://localhost:11434",
        model_name: str = "nomic-embed-text",
        batch_size: int = 32
    ):
        """Create manager with Ollama provider."""
        try:
            provider = OllamaEmbeddingProvider(base_url, model_name, batch_size)
            return cls(provider)
        except Exception as e:
            logging.error(f"Failed to create Ollama embedding manager: {e}")
            raise EmbeddingError(f"Ollama embedding manager creation failed: {e}") from e
    
    @classmethod
    def create_openai_manager(
        cls,
        api_key: str,
        model_name: str = "text-embedding-ada-002",
        batch_size: int = 32
    ):
        """Create manager with OpenAI provider."""
        try:
            provider = OpenAIEmbeddingProvider(api_key, model_name, batch_size)
            return cls(provider)
        except Exception as e:
            logging.error(f"Failed to create OpenAI embedding manager: {e}")
            raise EmbeddingError(f"OpenAI embedding manager creation failed: {e}") from e

    @classmethod
    def create_default_manager(
        cls,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: str = None,
        batch_size: int = 32
    ):
        """Create default embedding manager with HuggingFace provider for backward compatibility."""
        try:
            return cls.create_huggingface_manager(
                model_name=model_name,
                device=device,
                batch_size=batch_size
            )
        except Exception as e:
            logging.error(f"Failed to create default embedding manager: {e}")
            raise EmbeddingError(f"Default embedding manager creation failed: {e}") from e

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using the configured provider."""
        return await self.provider.generate_embeddings(texts)

    async def generate_chunk_embeddings(self, chunks: List[Chunk]) -> List[Chunk]:
        """Generate embeddings for chunks with content and context."""
        try:
            # Extract texts for content and context embeddings
            content_texts = [chunk.text for chunk in chunks]
            context_texts = []
            
            for chunk in chunks:
                # Create context text from surrounding content
                context_parts = []
                if hasattr(chunk, 'context_before') and chunk.context_before:
                    context_parts.append(chunk.context_before)
                context_parts.append(chunk.text)
                if hasattr(chunk, 'context_after') and chunk.context_after:
                    context_parts.append(chunk.context_after)
                context_texts.append(" ".join(context_parts))
            
            # Generate embeddings
            content_embeddings = await self.generate_embeddings(content_texts)
            context_embeddings = await self.generate_embeddings(context_texts)
            
            # Assign embeddings to chunks
            for chunk, content_emb, context_emb in zip(chunks, content_embeddings, context_embeddings):
                chunk.embedding = content_emb
                chunk.context_embedding = context_emb
            
            return chunks
            
        except Exception as e:
            raise EmbeddingError(f"Failed to generate chunk embeddings: {str(e)}")

    async def generate_query_embeddings(
        self,
        query: str,
        query_context: str = None
    ) -> Tuple[List[float], List[float]]:
        """Generate embeddings for query and context."""
        try:
            # Generate query embedding
            query_embeddings = await self.generate_embeddings([query])
            query_embedding = query_embeddings[0]
            
            # Generate context embedding if provided
            if query_context:
                context_embeddings = await self.generate_embeddings([query_context])
                context_embedding = context_embeddings[0]
            else:
                context_embedding = query_embedding
            
            return query_embedding, context_embedding
            
        except Exception as e:
            raise EmbeddingError(f"Failed to generate query embeddings: {str(e)}")

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return self.provider.get_embedding_dimension()

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return self.model_info.copy()

    async def migrate_from_old_model(
        self,
        old_manager: 'EnhancedEmbeddingManager',
        documents: List[Document],
        db: Session,
        storage_manager = None,
        progress_callback = None
    ) -> Dict[str, Any]:
        """Migrate embeddings from old model to new model."""
        try:
            start_time = datetime.utcnow()
            logger.info(f"Starting migration from {old_manager.model_info.get('model_name', 'unknown')} to {self.model_info['model_name']}")
            
            migrated_count = 0
            failed_count = 0
            total_documents = len(documents)
            
            results = {
                "migrated_documents": [],
                "failed_documents": [],
                "statistics": {}
            }
            
            for i, document in enumerate(documents):
                try:
                    # Delete old index
                    if storage_manager:
                        index_name = f"doc_{document.id}"
                        await storage_manager.delete_index(index_name, db=db)
                    
                    # Get document chunks
                    chunks = db.query(DocumentChunk).filter(
                        DocumentChunk.document_id == document.id
                    ).all()
                    
                    if not chunks:
                        logger.warning(f"No chunks found for document {document.id}")
                        continue
                    
                    # Generate new embeddings
                    content_texts = [chunk.text for chunk in chunks]
                    context_texts = []
                    
                    for chunk in chunks:
                        context_parts = []
                        if chunk.context_before:
                            context_parts.append(chunk.context_before)
                        context_parts.append(chunk.text)
                        if chunk.context_after:
                            context_parts.append(chunk.context_after)
                        context_texts.append(" ".join(context_parts))
                    
                    content_embeddings = await self.generate_embeddings(content_texts)
                    context_embeddings = await self.generate_embeddings(context_texts)
                    
                    # Store new embeddings
                    if storage_manager:
                        index_name = f"doc_{document.id}"
                        chunk_ids = [chunk.chunk_id for chunk in chunks]
                        
                        # Create new index with correct dimension
                        await storage_manager.create_index(
                            index_name=index_name,
                            embedding_dimension=self.get_embedding_dimension(),
                            faiss_index_type="HNSW",
                            user_id=document.user_id,
                            document_id=document.id,
                            db=db
                        )
                        
                        # Prepare metadata
                        metadata_list = []
                        for chunk in chunks:
                            metadata_list.append({
                                "chunk_id": chunk.chunk_id,
                                "document_id": document.id,
                                "chunk_index": chunk.chunk_index,
                                "text_length": chunk.text_length,
                                "page_number": chunk.page_number,
                                "section_title": chunk.section_title,
                                "embedding_model": self.model_info["model_name"]
                            })
                        
                        # Add vectors
                        await storage_manager.add_vectors(
                            index_name=index_name,
                            content_vectors=content_embeddings,
                            context_vectors=context_embeddings,
                            metadata_list=metadata_list,
                            chunk_ids=chunk_ids
                        )
                    
                    migrated_count += 1
                    results["migrated_documents"].append({
                        "document_id": document.id,
                        "filename": document.filename,
                        "chunks_migrated": len(chunks)
                    })
                    
                    # Progress callback
                    if progress_callback:
                        await progress_callback(i + 1, total_documents)
                        
                except Exception as e:
                    failed_count += 1
                    logger.error(f"Failed to migrate document {document.id}: {str(e)}")
                    results["failed_documents"].append({
                        "document_id": document.id,
                        "filename": document.filename,
                        "error": str(e)
                    })
            
            end_time = datetime.utcnow()
            processing_time = (end_time - start_time).total_seconds()
            
            results["statistics"] = {
                "total_documents": total_documents,
                "migrated_documents": migrated_count,
                "failed_documents": failed_count,
                "processing_time_seconds": processing_time,
                "old_model": old_manager.model_info.get('model_name', 'unknown'),
                "new_model": self.model_info["model_name"],
                "old_dimension": old_manager.get_embedding_dimension(),
                "new_dimension": self.get_embedding_dimension()
            }
            
            logger.info(f"Migration completed: {migrated_count}/{total_documents} documents migrated")
            return results
            
        except Exception as e:
            logger.error(f"Migration failed: {str(e)}")
            raise EmbeddingError(f"Migration failed: {str(e)}")

    async def health_check(self) -> Dict[str, Any]:
        """Check provider health."""
        try:
            # Test embedding generation with a simple text
            test_embeddings = await self.generate_embeddings(["Hello world"])
            
            return {
                "status": "healthy",
                "provider": self.model_info["provider"],
                "model": self.model_info["model_name"],
                "embedding_dimension": len(test_embeddings[0]),
                "test_passed": True
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": self.model_info["provider"],
                "model": self.model_info["model_name"],
                "error": str(e),
                "test_passed": False
            }

# Backward-compatible wrapper class
class EmbeddingManager:
    """Backward-compatible wrapper for EnhancedEmbeddingManager."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2", 
                 device: str = None, batch_size: int = 32):
        """Initialize with default HuggingFace provider for backward compatibility."""
        try:
            self._manager = EnhancedEmbeddingManager.create_default_manager(
                model_name=model_name, device=device, batch_size=batch_size
            )
        except Exception as e:
            logging.error(f"Failed to create default embedding manager: {e}")
            # Create a minimal fallback that doesn't crash
            self._manager = None
    
    def __getattr__(self, name):
        """Delegate all method calls to the internal manager."""
        if self._manager is None:
            raise EmbeddingError("Embedding manager not initialized properly")
        return getattr(self._manager, name)
    
    def __bool__(self):
        """Return True if manager is properly initialized."""
        return self._manager is not None