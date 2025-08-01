"""
Professional query processor for intelligent RAG search.
Separates semantic concepts from metadata and provides query intelligence.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
import unicodedata

logger = logging.getLogger(__name__)

@dataclass
class ProcessedQuery:
    """Represents a processed search query with extracted components."""
    primary_terms: List[str]  # Main semantic concepts
    secondary_terms: List[str]  # Metadata, dates, numbers
    exact_phrases: List[str]  # Quoted phrases
    excluded_terms: List[str]  # Terms to exclude
    original_query: str
    search_strategy: str
    confidence: float
    query_type: str
    
    def get_all_terms(self) -> List[str]:
        """Get all terms for fallback search."""
        return self.primary_terms + self.secondary_terms
    
    def get_weighted_terms(self) -> List[Tuple[str, float]]:
        """Get terms with their weights."""
        weighted = []
        for term in self.primary_terms:
            weighted.append((term, 0.8))
        for term in self.secondary_terms:
            weighted.append((term, 0.3))
        return weighted

class QueryProcessor:
    """
    Professional query processor that extracts semantic meaning from search queries.
    
    Features:
    - Separates concepts from metadata (dates, numbers, codes)
    - Handles quoted phrases and boolean operators
    - Normalizes and cleans query text
    - Provides search strategy recommendations
    """
    
    def __init__(self):
        # Patterns for different types of content
        self.date_patterns = [
            r'\b\d{1,2}\/\d{1,2}\/\d{2,4}\b',  # MM/DD/YYYY, M/D/YY
            r'\b\d{4}-\d{1,2}-\d{1,2}\b',      # YYYY-MM-DD
            r'\b\d{1,2}-\d{1,2}-\d{2,4}\b',    # MM-DD-YYYY
            r'\b\d{4}\b',                       # Just year
        ]
        
        self.code_patterns = [
            r'\b[A-Z]{2,}\d+[A-Z]*\b',         # ABC123, DOC456A
            r'\b\d+\/\d+\b',                   # 2/2025, 1/2024
            r'\b[A-Z]+\.\d+\b',                # REF.123
            r'\b#\w+\b',                       # #REF123
        ]
        
        self.number_patterns = [
            r'\b\d{3,}\b',                     # Numbers with 3+ digits
        ]
        
        # Stop words for better concept extraction
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
        }
        
        # Common business document terms that are semantic concepts
        self.concept_indicators = {
            'letter', 'document', 'report', 'memo', 'circular', 'notice', 'policy',
            'procedure', 'guideline', 'manual', 'contract', 'agreement', 'proposal',
            'invoice', 'receipt', 'statement', 'summary', 'analysis', 'review'
        }
    
    def process_query(self, query: str) -> ProcessedQuery:
        """
        Process a search query into semantic components.
        
        Args:
            query: Raw search query string
            
        Returns:
            ProcessedQuery with extracted components and search strategy
        """
        try:
            # Normalize the query
            normalized_query = self._normalize_query(query)
            
            # Extract quoted phrases first
            exact_phrases = self._extract_quoted_phrases(normalized_query)
            query_without_phrases = self._remove_quoted_phrases(normalized_query)
            
            # Extract different types of terms
            dates = self._extract_dates(query_without_phrases)
            codes = self._extract_codes(query_without_phrases)
            numbers = self._extract_numbers(query_without_phrases)
            
            # Remove extracted metadata from query for concept extraction
            concept_query = query_without_phrases
            for items in [dates, codes, numbers]:
                for item in items:
                    concept_query = concept_query.replace(item, ' ')
            
            # Extract semantic concepts
            concepts = self._extract_concepts(concept_query)
            
            # Determine search strategy and confidence
            search_strategy, confidence = self._determine_search_strategy(concepts, dates, codes)
            query_type = self._classify_query_type(concepts, dates, codes, exact_phrases)
            
            # Combine secondary terms
            secondary_terms = dates + codes + numbers
            
            processed = ProcessedQuery(
                primary_terms=concepts,
                secondary_terms=secondary_terms,
                exact_phrases=exact_phrases,
                excluded_terms=[],  # Could be extended for NOT operators
                original_query=query,
                search_strategy=search_strategy,
                confidence=confidence,
                query_type=query_type
            )
            
            logger.info(f"Processed query '{query}' -> concepts: {concepts}, metadata: {secondary_terms}")
            return processed
            
        except Exception as e:
            logger.error(f"Query processing failed for '{query}': {str(e)}")
            # Fallback to simple processing
            return self._create_fallback_query(query)
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query text for better processing."""
        # Convert to lowercase for processing (but preserve original case for phrases)
        normalized = query.strip()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Normalize unicode characters
        normalized = unicodedata.normalize('NFKD', normalized)
        
        return normalized
    
    def _extract_quoted_phrases(self, query: str) -> List[str]:
        """Extract exact phrases in quotes."""
        phrases = []
        
        # Find single and double quoted phrases
        quote_patterns = [
            r'"([^"]+)"',  # Double quotes
            r"'([^']+)'",  # Single quotes
        ]
        
        for pattern in quote_patterns:
            matches = re.findall(pattern, query)
            phrases.extend(matches)
        
        return [phrase.strip() for phrase in phrases if phrase.strip()]
    
    def _remove_quoted_phrases(self, query: str) -> str:
        """Remove quoted phrases from query for further processing."""
        # Remove double quoted phrases
        query = re.sub(r'"[^"]+"', ' ', query)
        # Remove single quoted phrases
        query = re.sub(r"'[^']+'", ' ', query)
        
        return re.sub(r'\s+', ' ', query).strip()
    
    def _extract_dates(self, query: str) -> List[str]:
        """Extract date patterns from query."""
        dates = []
        for pattern in self.date_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            dates.extend(matches)
        return dates
    
    def _extract_codes(self, query: str) -> List[str]:
        """Extract code patterns (document IDs, references) from query."""
        codes = []
        for pattern in self.code_patterns:
            matches = re.findall(pattern, query)
            codes.extend(matches)
        return codes
    
    def _extract_numbers(self, query: str) -> List[str]:
        """Extract significant numbers from query."""
        numbers = []
        for pattern in self.number_patterns:
            matches = re.findall(pattern, query)
            # Filter out dates and codes already extracted
            for match in matches:
                if not any(match in code for code in self._extract_codes(query)):
                    if not any(match in date for date in self._extract_dates(query)):
                        numbers.append(match)
        return numbers
    
    def _extract_concepts(self, query: str) -> List[str]:
        """Extract semantic concepts from query."""
        # Convert to lowercase for concept extraction
        query_lower = query.lower()
        
        # Split into words
        words = re.findall(r'\b\w+\b', query_lower)
        
        # Filter out stop words and short words
        concepts = []
        for word in words:
            if len(word) >= 2 and word not in self.stop_words:
                concepts.append(word)
        
        # Also extract multi-word concepts that are business terms
        concept_phrases = []
        for concept in self.concept_indicators:
            if concept in query_lower:
                # Look for multi-word combinations
                pattern = rf'\b\w*{concept}\w*\s+\w+\b|\b\w+\s+{concept}\w*\b'
                matches = re.findall(pattern, query_lower)
                concept_phrases.extend(matches)
        
        # Remove duplicates and return
        all_concepts = list(set(concepts + concept_phrases))
        
        # Prioritize concept indicators
        prioritized = []
        for concept in all_concepts:
            if any(indicator in concept for indicator in self.concept_indicators):
                prioritized.insert(0, concept)
            else:
                prioritized.append(concept)
        
        return prioritized
    
    def _determine_search_strategy(self, concepts: List[str], dates: List[str], codes: List[str]) -> Tuple[str, float]:
        """Determine the best search strategy based on query components."""
        confidence = 1.0
        
        # If we have strong concepts, use hybrid
        if len(concepts) >= 2:
            return "hybrid_weighted", 0.9
        
        # If we have one concept with metadata, use hybrid
        if len(concepts) >= 1 and (dates or codes):
            return "hybrid_weighted", 0.8
        
        # If mostly metadata, use text search
        if (dates or codes) and len(concepts) < 2:
            return "text_primary", 0.7
        
        # If only concepts, use semantic
        if len(concepts) >= 1:
            return "semantic_primary", 0.6
        
        # Fallback to hybrid
        return "hybrid_equal", 0.5
    
    def _classify_query_type(self, concepts: List[str], dates: List[str], codes: List[str], phrases: List[str]) -> str:
        """Classify the type of query for optimization."""
        if phrases:
            return "exact_phrase"
        elif codes and concepts:
            return "document_lookup"
        elif dates and concepts:
            return "temporal_concept"
        elif len(concepts) >= 2:
            return "multi_concept"
        elif len(concepts) == 1:
            return "single_concept"
        else:
            return "metadata_only"
    
    def _create_fallback_query(self, query: str) -> ProcessedQuery:
        """Create a fallback processed query when processing fails."""
        words = query.lower().split()
        
        return ProcessedQuery(
            primary_terms=words,
            secondary_terms=[],
            exact_phrases=[],
            excluded_terms=[],
            original_query=query,
            search_strategy="hybrid_equal",
            confidence=0.3,
            query_type="fallback"
        )
    
    def suggest_query_improvements(self, processed_query: ProcessedQuery) -> List[str]:
        """Suggest improvements for better search results."""
        suggestions = []
        
        if processed_query.confidence < 0.6:
            suggestions.append("Try using more specific terms or add quotes around exact phrases")
        
        if not processed_query.primary_terms:
            suggestions.append("Add descriptive terms about the document content you're looking for")
        
        if len(processed_query.primary_terms) == 1 and not processed_query.secondary_terms:
            suggestions.append("Add more context terms to improve search accuracy")
        
        return suggestions
    
    def expand_query(self, processed_query: ProcessedQuery) -> ProcessedQuery:
        """Expand query with synonyms and related terms."""
        # Simple synonym expansion for common business terms
        synonyms = {
            'letter': ['correspondence', 'communication', 'note'],
            'circular': ['notice', 'bulletin', 'announcement'],
            'policy': ['guideline', 'procedure', 'rule'],
            'report': ['document', 'analysis', 'summary'],
        }
        
        expanded_terms = processed_query.primary_terms.copy()
        for term in processed_query.primary_terms:
            if term in synonyms:
                # Add one synonym to avoid query explosion
                expanded_terms.append(synonyms[term][0])
        
        # Create new processed query with expanded terms
        expanded = ProcessedQuery(
            primary_terms=expanded_terms,
            secondary_terms=processed_query.secondary_terms,
            exact_phrases=processed_query.exact_phrases,
            excluded_terms=processed_query.excluded_terms,
            original_query=processed_query.original_query,
            search_strategy=processed_query.search_strategy,
            confidence=max(0.1, processed_query.confidence - 0.1),  # Slightly lower confidence
            query_type=f"{processed_query.query_type}_expanded"
        )
        
        return expanded


# Global query processor instance
_query_processor: Optional[QueryProcessor] = None

def get_query_processor() -> QueryProcessor:
    """Get the global query processor instance."""
    global _query_processor
    if _query_processor is None:
        _query_processor = QueryProcessor()
    return _query_processor