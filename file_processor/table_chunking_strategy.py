"""
Table-aware chunking strategy that preserves table structure and context.
Ensures reconstructed tables are not split across chunks and maintains meaningful context.
"""

import logging
import re
from typing import List, Dict, Optional, Tuple, Any, Union
from pathlib import Path
from dataclasses import dataclass, field

from .table_processor import CrossPageTableProcessor, ReconstructedTable, TableFragment
from .table_analyzer import TableAnalyzer, RAGTableRepresentation

# Optional import with graceful fallback
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logging.warning("Transformers not available for table-aware chunking")
    AutoTokenizer = None
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class TableAwareChunk:
    """Enhanced chunk with table awareness."""
    text: str
    start_idx: int
    end_idx: int
    metadata: Dict[str, Any]
    chunk_type: str = "text"  # "text", "table", "mixed"
    tables: List[ReconstructedTable] = field(default_factory=list)
    table_representations: List[RAGTableRepresentation] = field(default_factory=list)
    context_before: str = ""
    context_after: str = ""
    table_references: List[str] = field(default_factory=list)
    chunk_id: Optional[str] = None
    document_id: Optional[int] = None

@dataclass
class TableBoundary:
    """Information about table boundaries in document."""
    table_id: str
    start_position: int
    end_position: int
    pages: List[int]
    table: ReconstructedTable
    context_before: str = ""
    context_after: str = ""

class TableAwareChunkingStrategy:
    """Chunking strategy that preserves table structure and optimizes for RAG."""

    def __init__(self,
                 max_chunk_size: int = 1024,
                 table_context_size: int = 200,
                 min_chunk_size: int = 100,
                 tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize table-aware chunking strategy.

        Args:
            max_chunk_size: Maximum tokens per chunk
            table_context_size: Context size around tables
            min_chunk_size: Minimum tokens for a chunk
            tokenizer_name: Tokenizer for token counting
        """
        self.max_chunk_size = max_chunk_size
        self.table_context_size = table_context_size
        self.min_chunk_size = min_chunk_size

        # Initialize file manager for path resolution
        self.file_manager = None
        try:
            from utils.file_storage import get_file_manager
            self.file_manager = get_file_manager()
        except ImportError:
            logger.warning("File storage manager not available in table chunking strategy")

        # Initialize components with file manager
        self.table_processor = CrossPageTableProcessor(file_manager=self.file_manager)
        self.table_analyzer = TableAnalyzer()

        # Initialize tokenizer if available
        if TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            except Exception as e:
                logger.warning(f"Failed to load tokenizer {tokenizer_name}: {e}")
                self.tokenizer = None
        else:
            self.tokenizer = None

    def chunk_document(self,
                      file_path: Union[str, Path],
                      extracted_text: str,
                      metadata: Dict[str, Any]) -> List[TableAwareChunk]:
        """
        Create table-aware chunks from document.

        Args:
            file_path: Path to document file
            extracted_text: Pre-extracted text content
            metadata: Document metadata

        Returns:
            List of table-aware chunks
        """
        try:
            # Step 1: Extract and reconstruct tables
            tables = self.table_processor.process_document_tables(file_path)
            logger.info(f"Found {len(tables)} reconstructed tables")

            # Step 2: Create table representations for RAG
            table_representations = []
            for table in tables:
                analysis = self.table_analyzer.analyze_table(table)
                representations = self.table_analyzer.generate_rag_representation(table, analysis)
                table_representations.append(representations)

            # Step 3: Identify table boundaries in text
            table_boundaries = self._identify_table_boundaries(extracted_text, tables)

            # Step 4: Create table-aware chunks
            chunks = self._create_table_aware_chunks(
                extracted_text, tables, table_representations, table_boundaries, metadata
            )

            logger.info(f"Created {len(chunks)} table-aware chunks")
            return chunks

        except Exception as e:
            logger.error(f"Failed to create table-aware chunks: {e}")
            # Fallback to basic text chunking
            return self._fallback_text_chunking(extracted_text, metadata)

    def _identify_table_boundaries(self,
                                 text: str,
                                 tables: List[ReconstructedTable]) -> List[TableBoundary]:
        """Identify where tables appear in the extracted text."""
        boundaries = []

        for table in tables:
            # Try to find table content in text using various strategies
            table_positions = self._find_table_in_text(text, table)

            if table_positions:
                start_pos, end_pos = table_positions
                context_before = self._extract_context_before(text, start_pos)
                context_after = self._extract_context_after(text, end_pos)

                boundary = TableBoundary(
                    table_id=table.table_id,
                    start_position=start_pos,
                    end_position=end_pos,
                    pages=table.source_pages,
                    table=table,
                    context_before=context_before,
                    context_after=context_after
                )
                boundaries.append(boundary)

        # Sort boundaries by position
        boundaries.sort(key=lambda b: b.start_position)
        return boundaries

    def _find_table_in_text(self, text: str, table: ReconstructedTable) -> Optional[Tuple[int, int]]:
        """Find table content within extracted text."""
        # Strategy 1: Look for table headers
        if table.headers:
            header_pattern = r'\s+'.join(re.escape(header) for header in table.headers[:3])
            match = re.search(header_pattern, text, re.IGNORECASE)
            if match:
                start_pos = match.start()
                # Estimate end position based on table data
                estimated_length = sum(len(' '.join(row)) for row in table.data[:5])
                end_pos = min(start_pos + estimated_length + 200, len(text))
                return start_pos, end_pos

        # Strategy 2: Look for characteristic table data patterns
        if table.data:
            # Try first few data rows
            for row in table.data[:3]:
                if len(row) >= 2:
                    row_pattern = r'\s+'.join(re.escape(str(cell)) for cell in row[:3] if cell)
                    match = re.search(row_pattern, text, re.IGNORECASE)
                    if match:
                        start_pos = max(0, match.start() - 100)  # Include some context
                        estimated_length = sum(len(' '.join(row)) for row in table.data)
                        end_pos = min(start_pos + estimated_length + 300, len(text))
                        return start_pos, end_pos

        # Strategy 3: Look for table title if available
        if table.title:
            title_match = re.search(re.escape(table.title), text, re.IGNORECASE)
            if title_match:
                start_pos = title_match.start()
                # Estimate table size
                estimated_length = len(str(table.data)) if table.data else 500
                end_pos = min(start_pos + estimated_length, len(text))
                return start_pos, end_pos

        return None

    def _extract_context_before(self, text: str, position: int) -> str:
        """Extract context before table."""
        start = max(0, position - self.table_context_size)
        context = text[start:position].strip()

        # Try to end at sentence boundary
        last_sentence = context.rfind('.')
        if last_sentence > len(context) // 2:
            context = context[:last_sentence + 1]

        return context

    def _extract_context_after(self, text: str, position: int) -> str:
        """Extract context after table."""
        end = min(len(text), position + self.table_context_size)
        context = text[position:end].strip()

        # Try to end at sentence boundary
        first_sentence = context.find('.')
        if first_sentence > 0 and first_sentence < len(context) // 2:
            context = context[:first_sentence + 1]

        return context

    def _create_table_aware_chunks(self,
                                 text: str,
                                 tables: List[ReconstructedTable],
                                 representations: List[RAGTableRepresentation],
                                 boundaries: List[TableBoundary],
                                 metadata: Dict[str, Any]) -> List[TableAwareChunk]:
        """Create chunks that preserve table structure."""
        chunks = []
        current_position = 0
        chunk_counter = 0

        for boundary in boundaries:
            # Create text chunk before table (if any)
            if current_position < boundary.start_position:
                text_before = text[current_position:boundary.start_position].strip()
                if text_before and self._count_tokens(text_before) >= self.min_chunk_size:
                    text_chunks = self._split_text_chunk(text_before, metadata, chunk_counter)
                    chunks.extend(text_chunks)
                    chunk_counter += len(text_chunks)

            # Create table-specific chunk
            table_chunk = self._create_table_chunk(
                boundary, representations[boundaries.index(boundary)], metadata, chunk_counter
            )
            chunks.append(table_chunk)
            chunk_counter += 1

            current_position = boundary.end_position

        # Handle remaining text after last table
        if current_position < len(text):
            remaining_text = text[current_position:].strip()
            if remaining_text and self._count_tokens(remaining_text) >= self.min_chunk_size:
                text_chunks = self._split_text_chunk(remaining_text, metadata, chunk_counter)
                chunks.extend(text_chunks)

        return chunks

    def _create_table_chunk(self,
                          boundary: TableBoundary,
                          representation: RAGTableRepresentation,
                          metadata: Dict[str, Any],
                          chunk_id: int) -> TableAwareChunk:
        """Create a dedicated chunk for table content."""
        table = boundary.table

        # Combine multiple representations for comprehensive RAG content
        chunk_text = self._build_comprehensive_table_text(table, representation, boundary)

        chunk_metadata = metadata.copy()
        chunk_metadata.update({
            'table_id': table.table_id,
            'table_pages': table.source_pages,
            'table_rows': len(table.data),
            'table_columns': len(table.headers),
            'cross_page': table.cross_page,
            'table_confidence': table.confidence
        })

        return TableAwareChunk(
            text=chunk_text,
            start_idx=boundary.start_position,
            end_idx=boundary.end_position,
            metadata=chunk_metadata,
            chunk_type="table",
            tables=[table],
            table_representations=[representation],
            context_before=boundary.context_before,
            context_after=boundary.context_after,
            table_references=[table.table_id],
            chunk_id=f"table_{chunk_id}_{table.table_id}",
            document_id=metadata.get('document_id')
        )

    def _build_comprehensive_table_text(self,
                                       table: ReconstructedTable,
                                       representation: RAGTableRepresentation,
                                       boundary: TableBoundary) -> str:
        """Build comprehensive text representation for RAG."""
        components = []

        # Add context before
        if boundary.context_before:
            components.append(f"Context: {boundary.context_before}")

        # Add table title and description
        if table.title:
            components.append(f"Table: {table.title}")

        # Add semantic description
        if representation.semantic_description:
            components.append(f"Description: {representation.semantic_description}")

        # Add structured representation
        if representation.structured_json:
            components.append(f"Structure: {representation.structured_json}")

        # Add queryable format for search
        if representation.queryable_format:
            components.append(f"Content: {representation.queryable_format}")

        # Add metadata information
        if table.cross_page:
            components.append(f"Note: This table spans multiple pages ({', '.join(map(str, table.source_pages))})")

        # Add context after
        if boundary.context_after:
            components.append(f"Following context: {boundary.context_after}")

        return "\n\n".join(components)

    def _split_text_chunk(self,
                         text: str,
                         metadata: Dict[str, Any],
                         start_chunk_id: int) -> List[TableAwareChunk]:
        """Split text into appropriately sized chunks."""
        chunks = []
        sentences = self._split_into_sentences(text)

        current_chunk = []
        current_size = 0
        chunk_id = start_chunk_id

        for sentence in sentences:
            sentence_size = self._count_tokens(sentence)

            if current_size + sentence_size > self.max_chunk_size and current_chunk:
                # Create chunk
                chunk_text = " ".join(current_chunk)
                chunk = TableAwareChunk(
                    text=chunk_text,
                    start_idx=0,  # Relative positions would need document context
                    end_idx=len(chunk_text),
                    metadata=metadata.copy(),
                    chunk_type="text",
                    chunk_id=f"text_{chunk_id}",
                    document_id=metadata.get('document_id')
                )
                chunks.append(chunk)
                chunk_id += 1

                # Reset for next chunk
                current_chunk = []
                current_size = 0

            current_chunk.append(sentence)
            current_size += sentence_size

        # Handle remaining content
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunk = TableAwareChunk(
                text=chunk_text,
                start_idx=0,
                end_idx=len(chunk_text),
                metadata=metadata.copy(),
                chunk_type="text",
                chunk_id=f"text_{chunk_id}",
                document_id=metadata.get('document_id')
            )
            chunks.append(chunk)

        return chunks

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass

        # Fallback: estimate tokens as words * 1.3
        return int(len(text.split()) * 1.3)

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def _fallback_text_chunking(self, text: str, metadata: Dict[str, Any]) -> List[TableAwareChunk]:
        """Fallback to basic text chunking if table processing fails."""
        chunks = []
        sentences = self._split_into_sentences(text)

        current_chunk = []
        current_size = 0
        chunk_id = 0

        for sentence in sentences:
            sentence_size = self._count_tokens(sentence)

            if current_size + sentence_size > self.max_chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunk = TableAwareChunk(
                    text=chunk_text,
                    start_idx=0,
                    end_idx=len(chunk_text),
                    metadata=metadata.copy(),
                    chunk_type="text",
                    chunk_id=f"fallback_{chunk_id}",
                    document_id=metadata.get('document_id')
                )
                chunks.append(chunk)
                chunk_id += 1

                current_chunk = []
                current_size = 0

            current_chunk.append(sentence)
            current_size += sentence_size

        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunk = TableAwareChunk(
                text=chunk_text,
                start_idx=0,
                end_idx=len(chunk_text),
                metadata=metadata.copy(),
                chunk_type="text",
                chunk_id=f"fallback_{chunk_id}",
                document_id=metadata.get('document_id')
            )
            chunks.append(chunk)

        return chunks