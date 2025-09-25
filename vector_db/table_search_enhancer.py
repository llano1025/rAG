"""
Table-aware search enhancements for improved RAG retrieval of tabular data.
Provides specialized search capabilities for documents containing tables.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from database.models import DocumentChunk, Document
from api.schemas.search_schemas import SearchResult, SearchFilters, TableSearchFilter, TableSearchType

logger = logging.getLogger(__name__)


class TableSearchEnhancer:
    """Enhances search capabilities for table-aware content."""

    def __init__(self):
        """Initialize table search enhancer."""
        self.table_keywords = {
            'data', 'table', 'row', 'column', 'header', 'value', 'cell',
            'chart', 'graph', 'statistics', 'numbers', 'percentage',
            'total', 'sum', 'average', 'count', 'minimum', 'maximum'
        }

        # Patterns for detecting table-like queries
        self.table_query_patterns = [
            r'\b(table|data|rows?|columns?)\b',
            r'\b(statistics|stats|numbers?|values?)\b',
            r'\b(total|sum|average|mean|count)\b',
            r'\b(percentage|percent|%|ratio)\b',
            r'\b(minimum|maximum|min|max)\b',
            r'\b(comparison|compare|vs|versus)\b'
        ]

    def analyze_query_for_tables(self, query: str) -> Dict[str, Any]:
        """
        Analyze search query to determine if it's table-related.

        Args:
            query: Search query text

        Returns:
            Dictionary with table analysis results
        """
        query_lower = query.lower()

        # Count table-related keywords
        table_keyword_count = sum(1 for keyword in self.table_keywords if keyword in query_lower)

        # Check for table-specific patterns
        pattern_matches = []
        for pattern in self.table_query_patterns:
            matches = re.findall(pattern, query_lower)
            pattern_matches.extend(matches)

        # Determine table relevance score
        total_words = len(query.split())
        table_relevance_score = (table_keyword_count + len(pattern_matches)) / max(total_words, 1)

        # Detect specific table operations
        is_aggregation_query = any(word in query_lower for word in ['sum', 'total', 'average', 'count', 'max', 'min'])
        is_comparison_query = any(word in query_lower for word in ['compare', 'vs', 'versus', 'difference'])
        is_lookup_query = any(word in query_lower for word in ['find', 'show', 'list', 'what is'])

        return {
            'is_table_related': table_relevance_score > 0.2,
            'table_relevance_score': table_relevance_score,
            'table_keyword_count': table_keyword_count,
            'pattern_matches': pattern_matches,
            'is_aggregation_query': is_aggregation_query,
            'is_comparison_query': is_comparison_query,
            'is_lookup_query': is_lookup_query,
            'suggested_search_type': self._suggest_search_type(
                table_relevance_score, is_aggregation_query, is_comparison_query
            )
        }

    def _suggest_search_type(self, relevance_score: float, is_aggregation: bool, is_comparison: bool) -> str:
        """Suggest the best search type based on query analysis."""
        if relevance_score > 0.5:
            if is_aggregation or is_comparison:
                return TableSearchType.TABLE_CONTENT
            else:
                return TableSearchType.TABLE_HYBRID
        elif relevance_score > 0.2:
            return TableSearchType.TABLE_CONTEXT
        else:
            return "semantic"  # Default to semantic search

    async def enhance_table_search_results(
        self,
        results: List[SearchResult],
        query: str,
        table_filter: TableSearchFilter,
        db: Session
    ) -> List[SearchResult]:
        """
        Enhance search results with table-specific information and scoring.

        Args:
            results: Original search results
            query: Search query
            table_filter: Table-specific filters
            db: Database session

        Returns:
            Enhanced search results with table scoring
        """
        enhanced_results = []
        query_analysis = self.analyze_query_for_tables(query)

        for result in results:
            # Get chunk information
            chunk = db.query(DocumentChunk).filter(
                DocumentChunk.chunk_id == result.chunk_id
            ).first()

            if not chunk:
                enhanced_results.append(result)
                continue

            # Check if chunk contains table content
            table_info = self._analyze_chunk_for_tables(chunk, db)

            # Apply table-specific scoring boost
            enhanced_score = self._calculate_table_aware_score(
                result.score, table_info, query_analysis, table_filter
            )

            # Add table metadata to result
            enhanced_metadata = result.metadata.copy()
            enhanced_metadata.update({
                'table_info': table_info,
                'original_score': result.score,
                'table_relevance_boost': enhanced_score - result.score,
                'query_table_analysis': query_analysis
            })

            # Create enhanced result
            enhanced_result = SearchResult(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                text=result.text,
                score=enhanced_score,
                metadata=enhanced_metadata,
                document_metadata=result.document_metadata,
                highlight=result.highlight
            )

            enhanced_results.append(enhanced_result)

        # Sort by enhanced scores
        enhanced_results.sort(key=lambda r: r.score, reverse=True)

        return enhanced_results

    def _analyze_chunk_for_tables(self, chunk: DocumentChunk, db: Session) -> Dict[str, Any]:
        """
        Analyze a chunk to determine table content and properties.

        Args:
            chunk: Document chunk to analyze
            db: Database session

        Returns:
            Dictionary with table analysis results
        """
        # Check chunk metadata for table information
        metadata = chunk.get_metadata_dict() if hasattr(chunk, 'get_metadata_dict') else {}

        # Analyze text content for table patterns
        text_analysis = self._analyze_text_for_table_patterns(chunk.text)

        # Check if this chunk is marked as table content
        chunk_type = metadata.get('chunk_type', 'text')
        is_table_chunk = chunk_type in ['table', 'mixed']

        # Get table references from chunk
        table_references = metadata.get('table_references', [])

        # Get document-level table information
        document = db.query(Document).filter(Document.id == chunk.document_id).first()
        doc_metadata = document.get_metadata_dict() if document and hasattr(document, 'get_metadata_dict') else {}

        return {
            'is_table_chunk': is_table_chunk,
            'chunk_type': chunk_type,
            'table_references': table_references,
            'has_table_patterns': text_analysis['has_table_patterns'],
            'table_pattern_score': text_analysis['pattern_score'],
            'estimated_table_rows': text_analysis.get('estimated_rows', 0),
            'estimated_table_columns': text_analysis.get('estimated_columns', 0),
            'table_confidence': metadata.get('table_confidence', 0.0),
            'cross_page_tables': doc_metadata.get('cross_page_tables', False),
            'total_tables_in_document': doc_metadata.get('total_tables', 0)
        }

    def _analyze_text_for_table_patterns(self, text: str) -> Dict[str, Any]:
        """
        Analyze text content for table-like patterns.

        Args:
            text: Text content to analyze

        Returns:
            Dictionary with pattern analysis results
        """
        if not text:
            return {'has_table_patterns': False, 'pattern_score': 0.0}

        # Look for table-like structures in text
        lines = text.split('\n')

        # Count lines that look like table headers or data
        header_patterns = [
            r'^[A-Za-z\s]+\|[A-Za-z\s]+',  # Pipe-separated headers
            r'^[A-Za-z\s]+\t[A-Za-z\s]+',  # Tab-separated headers
            r'Table\s*\d*\s*[:\-]',         # Table captions
            r'Column\s*\d*\s*[:\-]',        # Column references
        ]

        data_patterns = [
            r'^\d+[\|\t]\d+',               # Numeric data with separators
            r'^\w+[\|\t]\w+[\|\t]\w+',      # Multi-column data
            r'^\s*\d+\.\d+\s*%',            # Percentages
            r'^\s*\$?\d+[\,\.]?\d*',        # Currency/numbers
        ]

        header_matches = 0
        data_matches = 0
        total_lines = len(lines)

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for header patterns
            for pattern in header_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    header_matches += 1
                    break

            # Check for data patterns
            for pattern in data_patterns:
                if re.search(pattern, line):
                    data_matches += 1
                    break

        # Estimate table dimensions
        estimated_columns = 0
        estimated_rows = max(header_matches, data_matches)

        # Try to estimate column count from separator patterns
        for line in lines[:5]:  # Check first few lines
            if '|' in line:
                estimated_columns = max(estimated_columns, line.count('|') + 1)
            elif '\t' in line:
                estimated_columns = max(estimated_columns, line.count('\t') + 1)

        # Calculate pattern score
        pattern_score = 0.0
        if total_lines > 0:
            pattern_score = (header_matches + data_matches) / total_lines

        return {
            'has_table_patterns': pattern_score > 0.1,
            'pattern_score': pattern_score,
            'header_matches': header_matches,
            'data_matches': data_matches,
            'estimated_rows': estimated_rows,
            'estimated_columns': estimated_columns
        }

    def _calculate_table_aware_score(
        self,
        original_score: float,
        table_info: Dict[str, Any],
        query_analysis: Dict[str, Any],
        table_filter: TableSearchFilter
    ) -> float:
        """
        Calculate enhanced score with table-awareness.

        Args:
            original_score: Original search score
            table_info: Table information for the chunk
            query_analysis: Query analysis results
            table_filter: Table search filters

        Returns:
            Enhanced score with table boosting
        """
        enhanced_score = original_score

        # Apply table relevance boost
        if query_analysis['is_table_related'] and table_info['is_table_chunk']:
            # Strong boost for table chunks when query is table-related
            table_boost = 0.3 * query_analysis['table_relevance_score']
            enhanced_score += table_boost

        # Boost for high-confidence table content
        if table_info['table_confidence'] > 0.7:
            confidence_boost = 0.2 * table_info['table_confidence']
            enhanced_score += confidence_boost

        # Boost for cross-page tables (more comprehensive content)
        if table_info.get('cross_page_tables', False):
            enhanced_score += 0.15

        # Apply query-specific boosts
        if query_analysis['is_aggregation_query'] and table_info['estimated_table_rows'] > 3:
            # Boost tables with multiple rows for aggregation queries
            enhanced_score += 0.25

        if query_analysis['is_comparison_query'] and table_info['estimated_table_columns'] > 2:
            # Boost multi-column tables for comparison queries
            enhanced_score += 0.2

        # Apply filter-based scoring
        if table_filter.table_only and not table_info['is_table_chunk']:
            # Penalize non-table content when table-only filter is active
            enhanced_score *= 0.5

        # Ensure score doesn't exceed reasonable bounds
        return min(enhanced_score, 1.0)

    def get_table_search_suggestions(self, query: str, db: Session) -> List[str]:
        """
        Generate search suggestions for table-related queries.

        Args:
            query: Original search query
            db: Database session

        Returns:
            List of suggested search refinements
        """
        suggestions = []
        query_analysis = self.analyze_query_for_tables(query)

        if query_analysis['is_table_related']:
            # Get common table-related terms from the database
            table_chunks = db.query(DocumentChunk).filter(
                DocumentChunk.metadata.ilike('%"chunk_type": "table"%')
            ).limit(100).all()

            # Extract common terms from table content
            common_table_terms = set()
            for chunk in table_chunks:
                if chunk.text:
                    words = re.findall(r'\b\w{3,}\b', chunk.text.lower())
                    common_table_terms.update(words[:10])  # Limit to avoid too many

            # Generate contextual suggestions
            base_terms = query.lower().split()

            # Add table-specific refinements
            if 'data' not in query.lower():
                suggestions.append(f"{query} data")

            if any(term in query.lower() for term in ['compare', 'comparison']):
                suggestions.append(f"{query} table comparison")

            if any(term in query.lower() for term in ['total', 'sum', 'average']):
                suggestions.append(f"{query} statistics")

            # Add common table terms if relevant
            relevant_terms = [term for term in common_table_terms if term not in base_terms][:3]
            for term in relevant_terms:
                suggestions.append(f"{query} {term}")

        return suggestions[:5]  # Limit to 5 suggestions