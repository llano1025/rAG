# from typing import List
# import torch
# from transformers import AutoModel, AutoTokenizer
# import numpy as np
# from .chunking import Chunk

# class EmbeddingError(Exception):
#     """Raised when embedding generation fails."""
#     pass

# class EmbeddingManager:
#     """Manages document embeddings with support for multiple models."""
    
#     def __init__(
#         self,
#         model_name: str = "sentence-transformers/all-mpnet-base-v2",
#         device: str = "cuda" if torch.cuda.is_available() else "cpu",
#         batch_size: int = 32
#     ):
#         self.model = AutoModel.from_pretrained(model_name)
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.device = device
#         self.batch_size = batch_size
#         self.model.to(device)
#         self.model.eval()

#     def generate_embeddings(self, chunks: List[Chunk]) -> List[Chunk]:
#         """Generate embeddings for a list of chunks."""
#         for i in range(0, len(chunks), self.batch_size):
#             batch = chunks[i:i + self.batch_size]
#             texts = [chunk.text for chunk in batch]
#             embeddings = self._batch_encode(texts)
            
#             for chunk, embedding in zip(batch, embeddings):
#                 chunk.embedding = embedding
        
#         return chunks

#     def _batch_encode(self, texts: List[str]) -> np.ndarray:
#         """Encode a batch of texts into embeddings."""
#         with torch.no_grad():
#             inputs = self.tokenizer(
#                 texts,
#                 padding=True,
#                 truncation=True,
#                 max_length=512,
#                 return_tensors="pt"
#             ).to(self.device)
            
#             outputs = self.model(**inputs)
#             embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
#             # Normalize embeddings
#             embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
#             return embeddings

from typing import List, Tuple
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
from .chunking import Chunk

class EmbeddingError(Exception):
    """Raised when embedding generation fails."""
    pass

class EmbeddingManager:
    """Manages document embeddings with contextual support."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 32
    ):
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device
        self.batch_size = batch_size
        self.model.to(device)
        self.model.eval()

    async def generate_embeddings(self, chunks: List[Chunk]) -> List[Chunk]:
        """Generate both content and context embeddings for chunks."""
        try:
            for i in range(0, len(chunks), self.batch_size):
                batch = chunks[i:i + self.batch_size]
                
                # Generate content embeddings
                texts = [chunk.text for chunk in batch]
                content_embeddings = self._batch_encode(texts)
                
                # Generate context embeddings
                context_texts = [
                    chunk.context_text if chunk.context_text 
                    else chunk.text for chunk in batch
                ]
                context_embeddings = self._batch_encode(context_texts)
                
                # Assign embeddings to chunks
                for chunk, content_emb, context_emb in zip(
                    batch, content_embeddings, context_embeddings
                ):
                    chunk.embedding = content_emb
                    chunk.context_embedding = context_emb
            
            return chunks
            
        except Exception as e:
            raise EmbeddingError(f"Failed to generate embeddings: {str(e)}")

    def _batch_encode(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of texts into embeddings."""
        with torch.no_grad():
            # Tokenize with special tokens
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate embeddings
            outputs = self.model(**inputs)
            
            # Use pooled output (CLS token) for sentence embedding
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            # Normalize embeddings
            embeddings = embeddings / np.linalg.norm(
                embeddings, axis=1, keepdims=True
            )
            
            return embeddings

    async def generate_query_embeddings(
        self,
        query: str,
        query_context: str = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate embeddings for query and its context."""
        try:
            # Generate query embedding
            query_embedding = self._batch_encode([query])[0]
            
            # Generate context embedding if provided
            if query_context:
                context_embedding = self._batch_encode([query_context])[0]
            else:
                context_embedding = query_embedding
            
            return query_embedding, context_embedding
            
        except Exception as e:
            raise EmbeddingError(
                f"Failed to generate query embeddings: {str(e)}"
            )