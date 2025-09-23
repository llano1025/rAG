"""
Advanced table processor for detecting and reconstructing cross-page tables.
Handles table fragmentation across page boundaries and creates unified table representations.
"""

import logging
import re
from typing import List, Dict, Tuple, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, field
import json
from difflib import SequenceMatcher

# Optional imports with graceful fallback
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    logging.warning("Pandas not available for advanced table processing")
    pd = None
    PANDAS_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    logging.warning("PyMuPDF not available for table spatial analysis")
    fitz = None
    PYMUPDF_AVAILABLE = False

try:
    import tabula
    TABULA_AVAILABLE = True
except ImportError:
    logging.warning("Tabula not available for table extraction")
    tabula = None
    TABULA_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class TableFragment:
    """Represents a table fragment from a single page."""
    page_number: int
    table_index: int  # Index within the page
    data: List[List[str]]
    headers: List[str]
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ReconstructedTable:
    """Represents a complete table reconstructed from fragments."""
    table_id: str
    title: Optional[str]
    headers: List[str]
    data: List[List[str]]
    source_pages: List[int]
    source_fragments: List[TableFragment]
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    cross_page: bool = False

@dataclass
class TableContext:
    """Context information surrounding a table."""
    preceding_text: str = ""
    following_text: str = ""
    section_title: str = ""
    references: List[str] = field(default_factory=list)

class CrossPageTableProcessor:
    """Advanced processor for detecting and reconstructing cross-page tables."""

    def __init__(self,
                 header_similarity_threshold: float = 0.8,
                 spatial_tolerance: float = 20.0,
                 min_table_rows: int = 2,
                 max_page_gap: int = 1,
                 file_manager=None):
        """
        Initialize the cross-page table processor.

        Args:
            header_similarity_threshold: Minimum similarity for matching headers
            spatial_tolerance: Pixel tolerance for spatial alignment
            min_table_rows: Minimum rows required to consider as table
            max_page_gap: Maximum page gap for table continuation
            file_manager: Optional file storage manager for path resolution
        """
        self.header_similarity_threshold = header_similarity_threshold
        self.spatial_tolerance = spatial_tolerance
        self.min_table_rows = min_table_rows
        self.max_page_gap = max_page_gap

        # Initialize file manager for path resolution
        self.file_manager = file_manager
        if file_manager is None:
            try:
                from utils.file_storage import get_file_manager
                self.file_manager = get_file_manager()
            except ImportError:
                logger.warning("File storage manager not available, using direct file paths")
                self.file_manager = None

    def process_document_tables(self, file_path: Union[str, Path]) -> List[ReconstructedTable]:
        """
        Process all tables in a document and reconstruct cross-page tables.

        Args:
            file_path: Path to the document file

        Returns:
            List of reconstructed tables
        """
        try:
            # Resolve file path if using file storage manager
            resolved_path = self._resolve_file_path(file_path)
            if resolved_path is None:
                logger.error(f"Could not resolve file path: {file_path}")
                return []

            # Extract raw table fragments from all pages
            fragments = self._extract_table_fragments(resolved_path)

            if not fragments:
                logger.debug(f"No table fragments found in {file_path}")
                return []

            # Group fragments into complete tables
            reconstructed_tables = self._reconstruct_tables(fragments)

            # Add context information
            for table in reconstructed_tables:
                context = self._extract_table_context(file_path, table)
                table.metadata['context'] = context

            logger.debug(f"Reconstructed {len(reconstructed_tables)} tables from {len(fragments)} fragments")
            return reconstructed_tables

        except Exception as e:
            logger.error(f"Failed to process document tables: {e}")
            return []

    def _resolve_file_path(self, file_path: Union[str, Path]) -> Optional[str]:
        """
        Resolve file path using file storage manager if available.

        Args:
            file_path: Original file path (could be relative storage path or absolute path)

        Returns:
            Absolute file path or None if file not found
        """
        try:
            file_path_str = str(file_path)

            # If path is absolute and exists, use it directly
            if Path(file_path_str).is_absolute() and Path(file_path_str).exists():
                logger.debug(f"Using absolute file path: {file_path_str}")
                return file_path_str

            # Try to resolve using file storage manager
            if self.file_manager:
                absolute_path = self.file_manager.get_absolute_path(file_path_str)
                if absolute_path:
                    logger.debug(f"Resolved storage path {file_path_str} to {absolute_path}")
                    return absolute_path
                else:
                    logger.warning(f"File storage manager could not resolve path: {file_path_str}")

            # If file manager is not available, try as relative path
            if Path(file_path_str).exists():
                logger.debug(f"Using relative file path: {file_path_str}")
                return file_path_str

            logger.error(f"Could not resolve file path: {file_path_str}")
            return None

        except Exception as e:
            logger.error(f"Error resolving file path {file_path}: {e}")
            return None

    def _extract_table_fragments(self, file_path: Union[str, Path]) -> List[TableFragment]:
        """Extract table fragments from each page of the document."""
        fragments = []

        # Validate file exists before attempting extraction
        if not Path(file_path).exists():
            logger.error(f"File not found for table extraction: {file_path}")
            return fragments

        # Check file size (avoid processing empty files)
        try:
            file_size = Path(file_path).stat().st_size
            if file_size == 0:
                logger.warning(f"File is empty, skipping table extraction: {file_path}")
                return fragments
            logger.debug(f"Processing file of size {file_size} bytes: {file_path}")
        except Exception as e:
            logger.error(f"Could not get file size for {file_path}: {e}")
            return fragments

        try:
            if TABULA_AVAILABLE:
                fragments.extend(self._extract_with_tabula(file_path))
                logger.debug("Extract table fragments with TABULA")

            if PYMUPDF_AVAILABLE and not fragments:
                # Fallback to PyMuPDF if tabula fails
                fragments.extend(self._extract_with_pymupdf(file_path))
                logger.debug("Extract table fragments with PYMUPDF")

        except FileNotFoundError as e:
            logger.error(f"File not found during table extraction: {file_path} - {e}")
        except PermissionError as e:
            logger.error(f"Permission denied accessing file: {file_path} - {e}")
        except Exception as e:
            logger.error(f"Table extraction failed for {file_path}: {e}")

        return fragments

    def _extract_with_tabula(self, file_path: Union[str, Path]) -> List[TableFragment]:
        """Extract table fragments using tabula-py."""
        fragments = []

        try:
            # Validate file is a PDF
            if not str(file_path).lower().endswith('.pdf'):
                logger.warning(f"Tabula requires PDF files, got: {file_path}")
                return fragments

            # Extract tables with page information
            tables = tabula.read_pdf(
                str(file_path),
                pages='all',
                multiple_tables=True,
                pandas_options={'header': None}  # Don't assume first row is header
            )

            # Get page information for each table
            logger.debug(f"Tabula extracted {len(tables) if tables else 0} tables from {file_path}")

            for page_num, page_tables in enumerate(tables):
                if isinstance(page_tables, list):
                    for table_idx, table_df in enumerate(page_tables):
                        if not table_df.empty and len(table_df) >= self.min_table_rows:
                            fragment = self._create_fragment_from_dataframe(
                                table_df, page_num + 1, table_idx
                            )
                            fragments.append(fragment)
                else:
                    # Single table
                    if not page_tables.empty and len(page_tables) >= self.min_table_rows:
                        fragment = self._create_fragment_from_dataframe(
                            page_tables, page_num + 1, 0
                        )
                        fragments.append(fragment)

        except Exception as e:
            logger.debug(f"Tabula extraction failed: {e}")

        return fragments

    def _extract_with_pymupdf(self, file_path: Union[str, Path]) -> List[TableFragment]:
        """Extract table fragments using PyMuPDF."""
        fragments = []

        try:
            with fitz.open(str(file_path)) as doc:
                for page_num, page in enumerate(doc):
                    page_tables = page.find_tables()

                    for table_idx, table in enumerate(page_tables):
                        try:
                            table_data = table.extract()
                            if table_data and len(table_data) >= self.min_table_rows:
                                bbox = table.bbox

                                # Clean and convert table data
                                cleaned_data = []
                                for row in table_data:
                                    if row and any(cell and str(cell).strip() for cell in row):
                                        clean_row = [str(cell).strip() if cell else "" for cell in row]
                                        cleaned_data.append(clean_row)

                                if cleaned_data:
                                    fragment = TableFragment(
                                        page_number=page_num + 1,
                                        table_index=table_idx,
                                        data=cleaned_data,
                                        headers=cleaned_data[0] if cleaned_data else [],
                                        bbox=bbox,
                                        confidence=0.8,  # Default confidence for PyMuPDF
                                        metadata={
                                            'extraction_method': 'pymupdf',
                                            'bbox': bbox
                                        }
                                    )
                                    fragments.append(fragment)

                        except Exception as e:
                            logger.debug(f"Failed to extract table {table_idx} from page {page_num + 1}: {e}")

        except Exception as e:
            logger.error(f"PyMuPDF extraction failed: {e}")

        return fragments

    def _create_fragment_from_dataframe(self, df, page_num: int, table_idx: int) -> TableFragment:
        """Create a TableFragment from a pandas DataFrame."""
        # Convert DataFrame to list of lists
        data = df.values.tolist()

        # Handle headers - assume first row might be headers if they look like headers
        headers = df.columns.tolist() if hasattr(df, 'columns') else []
        if not headers or all(str(h).startswith('Unnamed') for h in headers):
            # Use first row as headers if current headers are unnamed
            if data:
                headers = [str(cell) for cell in data[0]]
                data = data[1:]  # Remove header row from data

        # Clean data
        cleaned_data = []
        for row in data:
            clean_row = [str(cell).strip() if pd.notna(cell) else "" for cell in row]
            if any(clean_row):  # Only keep non-empty rows
                cleaned_data.append(clean_row)

        return TableFragment(
            page_number=page_num,
            table_index=table_idx,
            data=cleaned_data,
            headers=[str(h).strip() for h in headers],
            bbox=(0, 0, 0, 0),  # Tabula doesn't provide bbox info easily
            confidence=0.9,  # High confidence for tabula
            metadata={
                'extraction_method': 'tabula',
                'original_shape': df.shape if PANDAS_AVAILABLE else (len(data), len(data[0]) if data else 0)
            }
        )

    def _reconstruct_tables(self, fragments: List[TableFragment]) -> List[ReconstructedTable]:
        """Reconstruct complete tables from fragments."""
        if not fragments:
            return []

        reconstructed = []
        used_fragments = set()

        # Sort fragments by page number and table index
        sorted_fragments = sorted(fragments, key=lambda f: (f.page_number, f.table_index))

        for i, fragment in enumerate(sorted_fragments):
            if i in used_fragments:
                continue

            # Start a new table reconstruction
            table_fragments = [fragment]
            used_fragments.add(i)

            # Look for continuation fragments in subsequent pages
            for j in range(i + 1, len(sorted_fragments)):
                next_fragment = sorted_fragments[j]

                if j in used_fragments:
                    continue

                # Check if this fragment continues the current table
                if self._is_table_continuation(fragment, next_fragment, table_fragments):
                    table_fragments.append(next_fragment)
                    used_fragments.add(j)
                    fragment = next_fragment  # Update reference for next iteration

                # Stop if page gap is too large
                elif next_fragment.page_number - fragment.page_number > self.max_page_gap:
                    break

            # Reconstruct the table from fragments
            reconstructed_table = self._merge_table_fragments(table_fragments)
            if reconstructed_table:
                reconstructed.append(reconstructed_table)

        return reconstructed

    def _is_table_continuation(self,
                             current_fragment: TableFragment,
                             next_fragment: TableFragment,
                             existing_fragments: List[TableFragment]) -> bool:
        """Check if next_fragment is a continuation of the current table."""

        # Must be on consecutive or nearby pages
        page_diff = next_fragment.page_number - current_fragment.page_number
        if page_diff < 1 or page_diff > self.max_page_gap:
            return False

        # Compare headers similarity
        header_similarity = self._calculate_header_similarity(
            current_fragment.headers, next_fragment.headers
        )

        # Headers should be similar or next fragment should have no headers (continuation)
        if header_similarity >= self.header_similarity_threshold:
            return True

        # If next fragment has no clear headers, check column count compatibility
        if not next_fragment.headers or all(not h.strip() for h in next_fragment.headers):
            current_col_count = len(current_fragment.headers)
            next_col_count = len(next_fragment.data[0]) if next_fragment.data else 0

            # Column count should match
            if current_col_count == next_col_count:
                return True

        # Check spatial alignment if bbox information is available
        if (current_fragment.bbox != (0, 0, 0, 0) and
            next_fragment.bbox != (0, 0, 0, 0)):
            return self._check_spatial_alignment(current_fragment, next_fragment)

        return False

    def _calculate_header_similarity(self, headers1: List[str], headers2: List[str]) -> float:
        """Calculate similarity between two header lists."""
        if not headers1 or not headers2:
            return 0.0

        if len(headers1) != len(headers2):
            return 0.0

        similarities = []
        for h1, h2 in zip(headers1, headers2):
            if not h1.strip() or not h2.strip():
                continue

            similarity = SequenceMatcher(None, h1.lower().strip(), h2.lower().strip()).ratio()
            similarities.append(similarity)

        return sum(similarities) / len(similarities) if similarities else 0.0

    def _check_spatial_alignment(self, fragment1: TableFragment, fragment2: TableFragment) -> bool:
        """Check if two table fragments are spatially aligned."""
        bbox1 = fragment1.bbox
        bbox2 = fragment2.bbox

        # Check horizontal alignment (x-coordinates should be similar)
        x_diff = abs(bbox1[0] - bbox2[0])  # Left edge difference
        width_diff = abs((bbox1[2] - bbox1[0]) - (bbox2[2] - bbox2[0]))  # Width difference

        return x_diff <= self.spatial_tolerance and width_diff <= self.spatial_tolerance

    def _merge_table_fragments(self, fragments: List[TableFragment]) -> Optional[ReconstructedTable]:
        """Merge table fragments into a single reconstructed table."""
        if not fragments:
            return None

        # Use the first fragment's headers, or best headers found
        headers = self._determine_best_headers(fragments)

        # Merge all data rows
        all_data = []
        source_pages = []

        for fragment in fragments:
            source_pages.append(fragment.page_number)

            # Skip header rows in continuation fragments
            fragment_data = fragment.data
            if (len(fragments) > 1 and fragment != fragments[0] and
                self._looks_like_header_row(fragment_data[0] if fragment_data else [], headers)):
                fragment_data = fragment_data[1:]  # Skip duplicate header

            all_data.extend(fragment_data)

        if not all_data:
            return None

        # Create table ID
        table_id = f"table_{fragments[0].page_number}_{fragments[0].table_index}"
        if len(fragments) > 1:
            table_id += f"_to_{fragments[-1].page_number}"

        # Calculate confidence based on fragment confidences
        avg_confidence = sum(f.confidence for f in fragments) / len(fragments)

        return ReconstructedTable(
            table_id=table_id,
            title=self._generate_table_title(headers, all_data),
            headers=headers,
            data=all_data,
            source_pages=sorted(set(source_pages)),
            source_fragments=fragments,
            confidence=avg_confidence,
            cross_page=len(set(source_pages)) > 1,
            metadata={
                'fragment_count': len(fragments),
                'total_rows': len(all_data),
                'column_count': len(headers),
                'extraction_methods': list(set(f.metadata.get('extraction_method', 'unknown') for f in fragments))
            }
        )

    def _determine_best_headers(self, fragments: List[TableFragment]) -> List[str]:
        """Determine the best headers from available fragments."""
        if not fragments:
            return []

        # Try to find the most complete and meaningful headers
        best_headers = []
        best_score = -1

        for fragment in fragments:
            if not fragment.headers:
                continue

            # Score headers based on completeness and meaningfulness
            score = 0
            for header in fragment.headers:
                if header and header.strip():
                    score += 1
                    # Bonus for non-numeric headers (likely actual headers)
                    if not header.strip().isdigit():
                        score += 1

            if score > best_score:
                best_score = score
                best_headers = fragment.headers

        # If no good headers found, create generic ones
        if not best_headers and fragments[0].data:
            col_count = len(fragments[0].data[0]) if fragments[0].data else 0
            best_headers = [f"Column_{i+1}" for i in range(col_count)]

        return best_headers

    def _looks_like_header_row(self, row: List[str], expected_headers: List[str]) -> bool:
        """Check if a row looks like a header row."""
        if not row or not expected_headers:
            return False

        if len(row) != len(expected_headers):
            return False

        # Calculate similarity with expected headers
        similarities = []
        for cell, header in zip(row, expected_headers):
            if cell and header:
                similarity = SequenceMatcher(None, cell.lower().strip(), header.lower().strip()).ratio()
                similarities.append(similarity)

        if similarities:
            avg_similarity = sum(similarities) / len(similarities)
            return avg_similarity >= 0.7  # 70% similarity threshold

        return False

    def _generate_table_title(self, headers: List[str], data: List[List[str]]) -> Optional[str]:
        """Generate a descriptive title for the table based on its content."""
        if not headers:
            return None

        # Simple heuristic: use first few meaningful headers
        meaningful_headers = [h for h in headers[:3] if h and h.strip() and not h.strip().isdigit()]

        if meaningful_headers:
            if len(meaningful_headers) == 1:
                return f"{meaningful_headers[0]} Data"
            else:
                return f"{' and '.join(meaningful_headers)} Table"

        return "Data Table"

    def _extract_table_context(self, file_path: Union[str, Path], table: ReconstructedTable) -> TableContext:
        """Extract context information surrounding the table."""
        # This is a simplified implementation
        # In a full implementation, this would analyze the document text
        # around the table positions to extract relevant context

        context = TableContext()

        # For now, create basic context from table metadata
        if table.cross_page:
            context.references.append(f"Table spans pages {', '.join(map(str, table.source_pages))}")

        if table.title:
            context.section_title = table.title

        return context