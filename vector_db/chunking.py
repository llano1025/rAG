from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np
from transformers import AutoTokenizer
import re
from collections import deque

@dataclass
class Chunk:
    """Represents a document chunk with context."""
    text: str
    start_idx: int
    end_idx: int
    metadata: Dict
    context_text: Optional[str] = None
    embedding: Optional[np.ndarray] = None
    context_embedding: Optional[np.ndarray] = None
    neighbor_chunks: List[str] = None

class ChunkingError(Exception):
    """Raised when chunking operation fails."""
    pass

class AdaptiveChunker:
    """Implements context-aware adaptive document chunking."""
    
    def __init__(
        self,
        max_chunk_size: int = 512,
        overlap: int = 50,
        context_window: int = 200,
        tokenizer_name: str = "bert-base-uncased"
    ):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        self.context_window = context_window
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def chunk_document(self, text: str, metadata: Dict) -> List[Chunk]:
        """Split document into chunks with contextual information."""
        try:
            # Normalize text and split into semantic units
            text = self._normalize_text(text)
            sentences = self._split_into_sentences(text)
            
            chunks = []
            current_chunk = []
            current_size = 0
            start_idx = 0
            
            # Use deque to track recent chunks for context
            recent_chunks = deque(maxlen=3)
            
            for sentence in sentences:
                sentence_tokens = self.tokenizer.encode(sentence)
                sentence_size = len(sentence_tokens)
                
                if current_size + sentence_size > self.max_chunk_size and current_chunk:
                    # Create chunk with context
                    chunk_text = " ".join(current_chunk)
                    context = self._get_context(text, start_idx, start_idx + len(chunk_text))
                    
                    chunk = Chunk(
                        text=chunk_text,
                        start_idx=start_idx,
                        end_idx=start_idx + len(chunk_text),
                        metadata=metadata.copy(),
                        context_text=context,
                        neighbor_chunks=[c.text for c in recent_chunks]
                    )
                    
                    chunks.append(chunk)
                    recent_chunks.append(chunk)
                    
                    # Handle overlap
                    overlap_tokens = min(self.overlap, len(current_chunk[-1].split()))
                    current_chunk = current_chunk[-1].split()[-overlap_tokens:]
                    current_size = len(self.tokenizer.encode(" ".join(current_chunk)))
                    start_idx = start_idx + len(chunk_text) - len(" ".join(current_chunk))
                
                current_chunk.append(sentence)
                current_size += sentence_size
            
            # Handle remaining text
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                context = self._get_context(text, start_idx, start_idx + len(chunk_text))
                
                chunk = Chunk(
                    text=chunk_text,
                    start_idx=start_idx,
                    end_idx=start_idx + len(chunk_text),
                    metadata=metadata.copy(),
                    context_text=context,
                    neighbor_chunks=[c.text for c in recent_chunks]
                )
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            raise ChunkingError(f"Failed to chunk document: {str(e)}")

    def _get_context(self, text: str, start_idx: int, end_idx: int) -> str:
        """Get surrounding context for a chunk."""
        context_start = max(0, start_idx - self.context_window)
        context_end = min(len(text), end_idx + self.context_window)
        
        # Get text before and after the chunk
        before_context = text[context_start:start_idx].strip()
        after_context = text[end_idx:context_end].strip()
        
        return f"{before_context} [CHUNK] {after_context}"

    def _normalize_text(self, text: str) -> str:
        """Normalize text by removing extra whitespace and special characters."""
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex patterns."""
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text)
        return [s.strip() for s in sentences if s.strip()]