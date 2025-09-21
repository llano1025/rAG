# /vector_db/context_processor.py

from typing import Optional, List, Dict
import re
from dataclasses import dataclass

@dataclass
class ProcessedContext:
    text: str
    metadata: Dict
    weights: Dict[str, float]

class ContextProcessor:
    """Processes and enhances query context for better search."""
    
    def __init__(self):
        self.context_patterns = {
            'time': r'\b(today|yesterday|last\s+\w+)\b',
            'domain': r'\b(in|about|regarding)\s+(\w+)\b',
            'action': r'\b(how\s+to|what\s+is|define|explain)\b'
        }

    def process_context(
        self,
        query: str,
        context: Optional[str] = None
    ) -> str:
        """Process and enhance query context."""
        if not context:
            # Extract implicit context from query
            context = self._extract_implicit_context(query)
        
        # Combine explicit and implicit context
        combined_context = self._combine_context(query, context)
        
        # Clean and normalize
        cleaned_context = self._clean_context(combined_context)
        
        return cleaned_context

    def _extract_implicit_context(self, query: str) -> str:
        """Extract context implicitly from query."""
        contexts = []
        
        for context_type, pattern in self.context_patterns.items():
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                contexts.extend(matches)
        
        return " ".join(str(c) for c in contexts)

    def _combine_context(self, query: str, context: str) -> str:
        """Combine query and context information."""
        # Remove any duplicate information
        context_words = set(context.lower().split())
        query_words = set(query.lower().split())
        
        unique_context_words = context_words - query_words
        
        # Reconstruct context with unique information
        combined = f"{query} {' '.join(unique_context_words)}"
        return combined.strip()

    def _clean_context(self, context: str) -> str:
        """Clean and normalize context text."""
        # Remove extra whitespace
        context = re.sub(r'\s+', ' ', context)
        
        # Remove special characters
        context = re.sub(r'[^\w\s]', ' ', context)
        
        # Normalize to lowercase
        context = context.lower().strip()
        
        return context