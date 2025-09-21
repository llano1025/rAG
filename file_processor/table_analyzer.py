"""
Table analyzer for generating RAG-optimized representations of table data.
Creates multiple formats and semantic descriptions for enhanced search and understanding.
"""

import logging
import re
from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum
import statistics

# Check pandas availability without keeping reference
PANDAS_AVAILABLE = False
from .table_processor import ReconstructedTable

logger = logging.getLogger(__name__)

class ColumnType(Enum):
    """Types of table columns for semantic understanding."""
    TEXT = "text"
    NUMERIC = "numeric"
    CURRENCY = "currency"
    PERCENTAGE = "percentage"
    DATE = "date"
    BOOLEAN = "boolean"
    CATEGORICAL = "categorical"

@dataclass
class ColumnAnalysis:
    """Analysis results for a table column."""
    name: str
    type: ColumnType
    sample_values: List[str]
    unique_count: int
    null_count: int
    statistics: Dict[str, Any]
    patterns: List[str]

@dataclass
class TableAnalysis:
    """Comprehensive analysis of a table."""
    table_id: str
    summary: str
    insights: List[str]
    column_analyses: List[ColumnAnalysis]
    relationships: List[str]
    key_statistics: Dict[str, Any]

@dataclass
class RAGTableRepresentation:
    """Multiple representations of a table optimized for RAG."""
    table_id: str
    structured_json: Dict[str, Any]
    semantic_description: str
    queryable_format: str
    summary_text: str
    search_keywords: List[str]
    metadata: Dict[str, Any]

class TableAnalyzer:
    """Analyzes tables and generates RAG-optimized representations."""

    def __init__(self):
        self.currency_patterns = [
            r'\$[\d,]+\.?\d*',
            r'[\d,]+\.?\d*\s*(USD|EUR|GBP|JPY)',
            r'(USD|EUR|GBP|JPY)\s*[\d,]+\.?\d*'
        ]
        self.percentage_patterns = [
            r'[\d,]+\.?\d*\s*%',
            r'[\d,]+\.?\d*\s*percent'
        ]
        self.date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',
            r'[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}',
            r'Q[1-4]\s+\d{4}'
        ]

    def analyze_table(self, table: ReconstructedTable) -> TableAnalysis:
        """Perform comprehensive analysis of a table."""
        try:
            # Analyze each column
            column_analyses = []
            for i, header in enumerate(table.headers):
                column_data = [row[i] if i < len(row) else "" for row in table.data]
                analysis = self._analyze_column(header, column_data)
                column_analyses.append(analysis)

            # Generate insights
            insights = self._generate_insights(table, column_analyses)

            # Calculate key statistics
            key_statistics = self._calculate_key_statistics(table, column_analyses)

            # Generate summary
            summary = self._generate_table_summary(table, column_analyses, insights)

            # Identify relationships
            relationships = self._identify_relationships(column_analyses)

            return TableAnalysis(
                table_id=table.table_id,
                summary=summary,
                insights=insights,
                column_analyses=column_analyses,
                relationships=relationships,
                key_statistics=key_statistics
            )

        except Exception as e:
            logger.error(f"Failed to analyze table {table.table_id}: {e}")
            return TableAnalysis(
                table_id=table.table_id,
                summary=f"Table with {len(table.headers)} columns and {len(table.data)} rows",
                insights=[],
                column_analyses=[],
                relationships=[],
                key_statistics={}
            )

    def generate_rag_representation(self, table: ReconstructedTable, analysis: TableAnalysis) -> RAGTableRepresentation:
        """Generate multiple RAG-optimized representations of the table."""
        try:
            # 1. Structured JSON representation
            structured_json = self._create_structured_json(table, analysis)

            # 2. Semantic description
            semantic_description = self._create_semantic_description(table, analysis)

            # 3. Queryable format
            queryable_format = self._create_queryable_format(table, analysis)

            # 4. Summary text
            summary_text = self._create_summary_text(table, analysis)

            # 5. Search keywords
            search_keywords = self._extract_search_keywords(table, analysis)

            # 6. Metadata
            metadata = self._create_metadata(table, analysis)

            return RAGTableRepresentation(
                table_id=table.table_id,
                structured_json=structured_json,
                semantic_description=semantic_description,
                queryable_format=queryable_format,
                summary_text=summary_text,
                search_keywords=search_keywords,
                metadata=metadata
            )

        except Exception as e:
            logger.error(f"Failed to generate RAG representation for table {table.table_id}: {e}")
            return self._create_fallback_representation(table)

    def _analyze_column(self, header: str, data: List[str]) -> ColumnAnalysis:
        """Analyze a single column to determine its type and characteristics."""
        # Clean data
        clean_data = [str(item).strip() for item in data if item and str(item).strip()]

        if not clean_data:
            return ColumnAnalysis(
                name=header,
                type=ColumnType.TEXT,
                sample_values=[],
                unique_count=0,
                null_count=len(data),
                statistics={},
                patterns=[]
            )

        # Determine column type
        column_type = self._detect_column_type(clean_data)

        # Calculate statistics
        statistics = self._calculate_column_statistics(clean_data, column_type)

        # Extract patterns
        patterns = self._extract_patterns(clean_data, column_type)

        # Get sample values
        sample_values = clean_data[:5] if len(clean_data) <= 5 else clean_data[:3] + ["..."] + clean_data[-2:]

        return ColumnAnalysis(
            name=header,
            type=column_type,
            sample_values=sample_values,
            unique_count=len(set(clean_data)),
            null_count=len(data) - len(clean_data),
            statistics=statistics,
            patterns=patterns
        )

    def _detect_column_type(self, data: List[str]) -> ColumnType:
        """Detect the type of a column based on its data."""
        if not data:
            return ColumnType.TEXT

        # Check percentage
        percentage_count = sum(1 for item in data if any(re.search(pattern, item, re.IGNORECASE) for pattern in self.percentage_patterns))
        if percentage_count > len(data) * 0.7:
            return ColumnType.PERCENTAGE

        # Check currency
        currency_count = sum(1 for item in data if any(re.search(pattern, item, re.IGNORECASE) for pattern in self.currency_patterns))
        if currency_count > len(data) * 0.7:
            return ColumnType.CURRENCY

        # Check date
        date_count = sum(1 for item in data if any(re.search(pattern, item, re.IGNORECASE) for pattern in self.date_patterns))
        if date_count > len(data) * 0.7:
            return ColumnType.DATE

        # Check boolean
        boolean_values = {'true', 'false', 'yes', 'no', '1', '0', 'y', 'n'}
        boolean_count = sum(1 for item in data if item.lower() in boolean_values)
        if boolean_count > len(data) * 0.8:
            return ColumnType.BOOLEAN

        # Check numeric
        numeric_count = 0
        for item in data:
            # Remove common formatting characters
            clean_item = re.sub(r'[,\s$%]', '', item)
            try:
                float(clean_item)
                numeric_count += 1
            except ValueError:
                pass

        if numeric_count > len(data) * 0.8:
            return ColumnType.NUMERIC

        # Check categorical (limited unique values)
        unique_ratio = len(set(data)) / len(data)
        if unique_ratio < 0.5 and len(set(data)) < 20:
            return ColumnType.CATEGORICAL

        return ColumnType.TEXT

    def _calculate_column_statistics(self, data: List[str], column_type: ColumnType) -> Dict[str, Any]:
        """Calculate relevant statistics for a column."""
        stats = {}

        if column_type in [ColumnType.NUMERIC, ColumnType.CURRENCY, ColumnType.PERCENTAGE]:
            numeric_values = []
            for item in data:
                clean_item = re.sub(r'[,\s$%]', '', item)
                try:
                    numeric_values.append(float(clean_item))
                except ValueError:
                    pass

            if numeric_values:
                stats.update({
                    'min': min(numeric_values),
                    'max': max(numeric_values),
                    'mean': statistics.mean(numeric_values),
                    'median': statistics.median(numeric_values),
                    'std_dev': statistics.stdev(numeric_values) if len(numeric_values) > 1 else 0
                })

        elif column_type == ColumnType.CATEGORICAL:
            value_counts = {}
            for item in data:
                value_counts[item] = value_counts.get(item, 0) + 1
            stats['value_counts'] = dict(sorted(value_counts.items(), key=lambda x: x[1], reverse=True))

        elif column_type == ColumnType.TEXT:
            lengths = [len(item) for item in data]
            stats.update({
                'avg_length': sum(lengths) / len(lengths),
                'min_length': min(lengths),
                'max_length': max(lengths)
            })

        return stats

    def _extract_patterns(self, data: List[str], column_type: ColumnType) -> List[str]:
        """Extract common patterns from column data."""
        patterns = []

        if column_type == ColumnType.DATE:
            for pattern in self.date_patterns:
                matches = [item for item in data if re.search(pattern, item)]
                if matches:
                    patterns.append(f"Date format: {pattern} (e.g., {matches[0]})")

        elif column_type == ColumnType.CURRENCY:
            for pattern in self.currency_patterns:
                matches = [item for item in data if re.search(pattern, item)]
                if matches:
                    patterns.append(f"Currency format: {matches[0]}")

        elif column_type == ColumnType.TEXT:
            # Look for common text patterns
            if all(len(item.split()) == 1 for item in data[:5]):
                patterns.append("Single word entries")
            elif all(item.isupper() for item in data[:5]):
                patterns.append("Uppercase text")
            elif all(item.istitle() for item in data[:5]):
                patterns.append("Title case text")

        return patterns

    def _generate_insights(self, table: ReconstructedTable, column_analyses: List[ColumnAnalysis]) -> List[str]:
        """Generate insights about the table."""
        insights = []

        # Cross-page insight
        if table.cross_page:
            insights.append(f"This table spans {len(table.source_pages)} pages ({', '.join(map(str, table.source_pages))})")

        # Size insights
        insights.append(f"Contains {len(table.data)} rows and {len(table.headers)} columns")

        # Column type insights
        type_counts = {}
        for analysis in column_analyses:
            type_name = analysis.type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1

        for col_type, count in type_counts.items():
            if count > 1:
                insights.append(f"Has {count} {col_type} columns")

        # Numeric insights
        numeric_columns = [a for a in column_analyses if a.type in [ColumnType.NUMERIC, ColumnType.CURRENCY, ColumnType.PERCENTAGE]]
        if numeric_columns:
            for col in numeric_columns:
                if col.statistics:
                    min_val = col.statistics.get('min')
                    max_val = col.statistics.get('max')
                    if min_val is not None and max_val is not None:
                        insights.append(f"{col.name} ranges from {min_val} to {max_val}")

        return insights

    def _calculate_key_statistics(self, table: ReconstructedTable, column_analyses: List[ColumnAnalysis]) -> Dict[str, Any]:
        """Calculate key statistics about the table."""
        stats = {
            'total_rows': len(table.data),
            'total_columns': len(table.headers),
            'cross_page': table.cross_page,
            'source_pages': table.source_pages,
            'confidence': table.confidence
        }

        # Column type distribution
        type_distribution = {}
        for analysis in column_analyses:
            type_name = analysis.type.value
            type_distribution[type_name] = type_distribution.get(type_name, 0) + 1

        stats['column_types'] = type_distribution

        # Data completeness
        total_cells = len(table.data) * len(table.headers)
        null_cells = sum(analysis.null_count for analysis in column_analyses)
        stats['completeness'] = (total_cells - null_cells) / total_cells if total_cells > 0 else 0

        return stats

    def _generate_table_summary(self, table: ReconstructedTable, column_analyses: List[ColumnAnalysis], insights: List[str]) -> str:
        """Generate a comprehensive summary of the table."""
        summary_parts = []

        # Title and basic info
        if table.title:
            summary_parts.append(f"Table: {table.title}")
        else:
            summary_parts.append(f"Data table with {len(table.headers)} columns")

        # Cross-page info
        if table.cross_page:
            summary_parts.append(f"This table spans pages {', '.join(map(str, table.source_pages))}")

        # Column descriptions
        col_descriptions = []
        for analysis in column_analyses:
            desc = f"{analysis.name} ({analysis.type.value})"
            if analysis.sample_values:
                sample = ", ".join(str(v) for v in analysis.sample_values[:3])
                desc += f" - examples: {sample}"
            col_descriptions.append(desc)

        summary_parts.append(f"Columns: {'; '.join(col_descriptions)}")

        # Key insights
        if insights:
            summary_parts.append(f"Key insights: {'; '.join(insights[:3])}")

        return ". ".join(summary_parts) + "."

    def _identify_relationships(self, column_analyses: List[ColumnAnalysis]) -> List[str]:
        """Identify potential relationships between columns."""
        relationships = []

        # Look for potential key columns
        for analysis in column_analyses:
            if analysis.unique_count == len(analysis.sample_values) and analysis.unique_count > 1:
                relationships.append(f"{analysis.name} appears to be a unique identifier")

        # Look for paired columns (e.g., start/end dates, min/max values)
        column_names = [a.name.lower() for a in column_analyses]
        pairs = [
            ('start', 'end'),
            ('begin', 'finish'),
            ('min', 'max'),
            ('from', 'to'),
            ('before', 'after')
        ]

        for pair in pairs:
            if pair[0] in column_names and pair[1] in column_names:
                relationships.append(f"Contains {pair[0]} and {pair[1]} columns indicating a range or period")

        return relationships

    def _create_structured_json(self, table: ReconstructedTable, analysis: TableAnalysis) -> Dict[str, Any]:
        """Create structured JSON representation."""
        return {
            "type": "table",
            "id": table.table_id,
            "title": table.title,
            "headers": table.headers,
            "rows": table.data,
            "metadata": {
                "source_pages": table.source_pages,
                "column_types": {col.name: col.type.value for col in analysis.column_analyses},
                "cross_page": table.cross_page,
                "confidence": table.confidence,
                "row_count": len(table.data),
                "column_count": len(table.headers),
                "completeness": analysis.key_statistics.get('completeness', 0)
            },
            "analysis": {
                "insights": analysis.insights,
                "relationships": analysis.relationships,
                "summary": analysis.summary
            }
        }

    def _create_semantic_description(self, table: ReconstructedTable, analysis: TableAnalysis) -> str:
        """Create semantic text description."""
        description_parts = []

        # Title and context
        if table.title:
            description_parts.append(f"Table: {table.title}")

        if table.cross_page:
            description_parts.append(f"(spans pages {', '.join(map(str, table.source_pages))})")

        # Structure description
        description_parts.append(f"This table contains {len(table.data)} rows and {len(table.headers)} columns.")

        # Column descriptions
        col_descriptions = []
        for col in analysis.column_analyses:
            desc = f"{col.name} ({col.type.value})"
            if col.statistics and col.type in [ColumnType.NUMERIC, ColumnType.CURRENCY]:
                min_val = col.statistics.get('min')
                max_val = col.statistics.get('max')
                if min_val is not None and max_val is not None:
                    desc += f" ranging from {min_val} to {max_val}"
            elif col.type == ColumnType.CATEGORICAL and col.statistics.get('value_counts'):
                top_values = list(col.statistics['value_counts'].keys())[:3]
                desc += f" with values like {', '.join(top_values)}"
            col_descriptions.append(desc)

        description_parts.append(f"Columns include: {'; '.join(col_descriptions)}.")

        # Key insights
        if analysis.insights:
            description_parts.append(f"Key insights: {' '.join(analysis.insights)}.")

        return " ".join(description_parts)

    def _create_queryable_format(self, table: ReconstructedTable, analysis: TableAnalysis) -> str:
        """Create queryable text format."""
        lines = []

        # Header
        lines.append(f"Table: {table.title or 'Data Table'}")
        if table.cross_page:
            lines.append(f"Source: Pages {', '.join(map(str, table.source_pages))}")

        # Column information
        for col in analysis.column_analyses:
            line = f"- {col.name} ({col.type.value})"
            if col.sample_values:
                examples = [str(v) for v in col.sample_values[:3] if str(v) != "..."]
                line += f": {', '.join(examples)}"
            lines.append(line)

        # Summary statistics
        if analysis.key_statistics:
            stats = analysis.key_statistics
            lines.append(f"Statistics: {stats.get('total_rows', 0)} rows, {stats.get('total_columns', 0)} columns")

        # Full data (formatted for search)
        lines.append("Data:")
        for i, row in enumerate(table.data[:10]):  # Limit to first 10 rows for search
            row_text = " | ".join(str(cell) for cell in row)
            lines.append(f"Row {i+1}: {row_text}")

        if len(table.data) > 10:
            lines.append(f"... and {len(table.data) - 10} more rows")

        return "\n".join(lines)

    def _create_summary_text(self, table: ReconstructedTable, analysis: TableAnalysis) -> str:
        """Create concise summary text."""
        _ = table  # Unused parameter, kept for interface consistency
        return analysis.summary

    def _extract_search_keywords(self, table: ReconstructedTable, analysis: TableAnalysis) -> List[str]:
        """Extract keywords for search optimization."""
        keywords = []

        # Table title words
        if table.title:
            keywords.extend(table.title.lower().split())

        # Column names
        keywords.extend([header.lower() for header in table.headers])

        # Column types
        keywords.extend([col.type.value for col in analysis.column_analyses])

        # Sample values (for categorical data)
        for col in analysis.column_analyses:
            if col.type == ColumnType.CATEGORICAL and col.sample_values:
                keywords.extend([str(v).lower() for v in col.sample_values[:5] if str(v) != "..."])

        # Remove duplicates and common words
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an'}
        keywords = list(set(k for k in keywords if k and len(k) > 2 and k not in stop_words))

        return keywords

    def _create_metadata(self, table: ReconstructedTable, analysis: TableAnalysis) -> Dict[str, Any]:
        """Create metadata for the table."""
        return {
            'processing_timestamp': None,  # Would be set by caller
            'extraction_confidence': table.confidence,
            'analysis_version': '1.0',
            'cross_page': table.cross_page,
            'source_pages': table.source_pages,
            'data_quality_score': analysis.key_statistics.get('completeness', 0),
            'column_count': len(table.headers),
            'row_count': len(table.data),
            'has_numeric_data': any(col.type in [ColumnType.NUMERIC, ColumnType.CURRENCY, ColumnType.PERCENTAGE]
                                   for col in analysis.column_analyses),
            'has_categorical_data': any(col.type == ColumnType.CATEGORICAL for col in analysis.column_analyses)
        }

    def _create_fallback_representation(self, table: ReconstructedTable) -> RAGTableRepresentation:
        """Create a fallback representation when analysis fails."""
        return RAGTableRepresentation(
            table_id=table.table_id,
            structured_json={
                "type": "table",
                "id": table.table_id,
                "headers": table.headers,
                "rows": table.data,
                "metadata": {"cross_page": table.cross_page}
            },
            semantic_description=f"Table with {len(table.headers)} columns and {len(table.data)} rows",
            queryable_format=f"Table headers: {', '.join(table.headers)}",
            summary_text=f"Data table ({len(table.data)} rows, {len(table.headers)} columns)",
            search_keywords=table.headers + [table.table_id],
            metadata={
                'fallback': True,
                'cross_page': table.cross_page
            }
        )