"""
Duplicate Detection System for RAG Documents
Detects and manages duplicate documents using multiple similarity methods.
"""

import hashlib
import logging
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import numpy as np
from difflib import SequenceMatcher
import asyncio
import aiofiles
from pathlib import Path
import json
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import imagehash
from PIL import Image
import io

logger = logging.getLogger(__name__)


class DuplicateType(Enum):
    """Types of duplicate detection methods"""
    EXACT_HASH = "exact_hash"
    CONTENT_HASH = "content_hash"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    STRUCTURAL_SIMILARITY = "structural_similarity"
    IMAGE_PERCEPTUAL = "image_perceptual"


@dataclass
class DuplicateMatch:
    """Represents a duplicate match between documents"""
    document_id_1: str
    document_id_2: str
    similarity_score: float
    duplicate_type: DuplicateType
    confidence: float
    metadata: Dict = None
    detected_at: datetime = None

    def __post_init__(self):
        if self.detected_at is None:
            self.detected_at = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class DocumentFingerprint:
    """Document fingerprint for duplicate detection"""
    document_id: str
    file_hash: str
    content_hash: str
    structural_hash: str
    tfidf_vector: Optional[np.ndarray] = None
    image_hash: Optional[str] = None
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DuplicateDetector:
    """Advanced duplicate detection system"""

    def __init__(self, config: Dict = None):
        self.config = config or self._get_default_config()
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.config['tfidf_max_features'],
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.document_fingerprints: Dict[str, DocumentFingerprint] = {}
        self.duplicate_matches: List[DuplicateMatch] = []
        self.similarity_thresholds = self.config['similarity_thresholds']

    def _get_default_config(self) -> Dict:
        """Get default configuration for duplicate detection"""
        return {
            'similarity_thresholds': {
                'exact_hash': 1.0,
                'content_similarity': 0.95,
                'semantic_similarity': 0.85,
                'structural_similarity': 0.90,
                'image_perceptual': 0.90
            },
            'tfidf_max_features': 10000,
            'enable_fuzzy_matching': True,
            'fuzzy_threshold': 0.8,
            'batch_size': 100,
            'cache_fingerprints': True,
            'fingerprint_cache_path': 'cache/document_fingerprints.json'
        }

    async def generate_fingerprint(self, document_id: str, file_path: str, 
                                 content: str, file_type: str) -> DocumentFingerprint:
        """Generate comprehensive fingerprint for a document"""
        try:
            # File hash (exact duplicate detection)
            file_hash = await self._calculate_file_hash(file_path)
            
            # Content hash (near-exact duplicate detection)
            content_hash = self._calculate_content_hash(content)
            
            # Structural hash (layout and structure similarity)
            structural_hash = self._calculate_structural_hash(content, file_type)
            
            # TF-IDF vector for semantic similarity
            tfidf_vector = None
            if content and len(content.strip()) > 0:
                tfidf_vector = self._calculate_tfidf_vector(content)
            
            # Image perceptual hash for image files
            image_hash = None
            if file_type.lower() in ['png', 'jpg', 'jpeg', 'gif', 'bmp']:
                image_hash = await self._calculate_image_hash(file_path)

            fingerprint = DocumentFingerprint(
                document_id=document_id,
                file_hash=file_hash,
                content_hash=content_hash,
                structural_hash=structural_hash,
                tfidf_vector=tfidf_vector,
                image_hash=image_hash,
                metadata={
                    'file_type': file_type,
                    'content_length': len(content),
                    'generated_at': datetime.utcnow().isoformat()
                }
            )

            # Cache fingerprint
            self.document_fingerprints[document_id] = fingerprint
            
            if self.config['cache_fingerprints']:
                await self._cache_fingerprint(fingerprint)

            return fingerprint

        except Exception as e:
            logger.error(f"Error generating fingerprint for document {document_id}: {e}")
            raise

    async def detect_duplicates(self, document_id: str, 
                              fingerprint: DocumentFingerprint = None) -> List[DuplicateMatch]:
        """Detect duplicates for a given document"""
        if fingerprint is None:
            fingerprint = self.document_fingerprints.get(document_id)
            if not fingerprint:
                raise ValueError(f"No fingerprint found for document {document_id}")

        duplicates = []
        
        # Compare with all existing documents
        for existing_id, existing_fingerprint in self.document_fingerprints.items():
            if existing_id == document_id:
                continue
                
            matches = await self._compare_fingerprints(fingerprint, existing_fingerprint)
            duplicates.extend(matches)

        # Store found duplicates
        self.duplicate_matches.extend(duplicates)
        
        return duplicates

    async def _compare_fingerprints(self, fp1: DocumentFingerprint, 
                                  fp2: DocumentFingerprint) -> List[DuplicateMatch]:
        """Compare two document fingerprints for duplicates"""
        matches = []

        # Exact file hash match
        if fp1.file_hash == fp2.file_hash:
            matches.append(DuplicateMatch(
                document_id_1=fp1.document_id,
                document_id_2=fp2.document_id,
                similarity_score=1.0,
                duplicate_type=DuplicateType.EXACT_HASH,
                confidence=1.0,
                metadata={'method': 'file_hash'}
            ))

        # Content hash similarity
        content_similarity = self._calculate_hash_similarity(fp1.content_hash, fp2.content_hash)
        if content_similarity >= self.similarity_thresholds['content_similarity']:
            matches.append(DuplicateMatch(
                document_id_1=fp1.document_id,
                document_id_2=fp2.document_id,
                similarity_score=content_similarity,
                duplicate_type=DuplicateType.CONTENT_HASH,
                confidence=content_similarity,
                metadata={'method': 'content_hash'}
            ))

        # Semantic similarity using TF-IDF
        if fp1.tfidf_vector is not None and fp2.tfidf_vector is not None:
            semantic_similarity = self._calculate_cosine_similarity(
                fp1.tfidf_vector, fp2.tfidf_vector
            )
            if semantic_similarity >= self.similarity_thresholds['semantic_similarity']:
                matches.append(DuplicateMatch(
                    document_id_1=fp1.document_id,
                    document_id_2=fp2.document_id,
                    similarity_score=semantic_similarity,
                    duplicate_type=DuplicateType.SEMANTIC_SIMILARITY,
                    confidence=semantic_similarity,
                    metadata={'method': 'tfidf_cosine'}
                ))

        # Structural similarity
        structural_similarity = self._calculate_hash_similarity(
            fp1.structural_hash, fp2.structural_hash
        )
        if structural_similarity >= self.similarity_thresholds['structural_similarity']:
            matches.append(DuplicateMatch(
                document_id_1=fp1.document_id,
                document_id_2=fp2.document_id,
                similarity_score=structural_similarity,
                duplicate_type=DuplicateType.STRUCTURAL_SIMILARITY,
                confidence=structural_similarity,
                metadata={'method': 'structural_hash'}
            ))

        # Image perceptual hash
        if fp1.image_hash and fp2.image_hash:
            image_similarity = self._calculate_image_similarity(fp1.image_hash, fp2.image_hash)
            if image_similarity >= self.similarity_thresholds['image_perceptual']:
                matches.append(DuplicateMatch(
                    document_id_1=fp1.document_id,
                    document_id_2=fp2.document_id,
                    similarity_score=image_similarity,
                    duplicate_type=DuplicateType.IMAGE_PERCEPTUAL,
                    confidence=image_similarity,
                    metadata={'method': 'perceptual_hash'}
                ))

        return matches

    async def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file"""
        hash_sha256 = hashlib.sha256()
        async with aiofiles.open(file_path, 'rb') as f:
            while chunk := await f.read(8192):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def _calculate_content_hash(self, content: str) -> str:
        """Calculate hash of normalized content"""
        # Normalize content: remove extra whitespace, convert to lowercase
        normalized = ' '.join(content.lower().split())
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()

    def _calculate_structural_hash(self, content: str, file_type: str) -> str:
        """Calculate hash based on document structure"""
        structural_features = []
        
        # Line count
        lines = content.split('\n')
        structural_features.append(f"lines:{len(lines)}")
        
        # Paragraph count (empty lines as separators)
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        structural_features.append(f"paragraphs:{len(paragraphs)}")
        
        # Average line length
        avg_line_length = sum(len(line) for line in lines) / len(lines) if lines else 0
        structural_features.append(f"avg_line_length:{int(avg_line_length)}")
        
        # Word count
        words = content.split()
        structural_features.append(f"words:{len(words)}")
        
        # Character distribution (letters, digits, punctuation)
        letters = sum(1 for c in content if c.isalpha())
        digits = sum(1 for c in content if c.isdigit())
        punctuation = sum(1 for c in content if c in '.,!?;:"()[]{}')
        total_chars = len(content)
        
        if total_chars > 0:
            structural_features.extend([
                f"letters_ratio:{letters/total_chars:.3f}",
                f"digits_ratio:{digits/total_chars:.3f}",
                f"punct_ratio:{punctuation/total_chars:.3f}"
            ])

        structural_string = '|'.join(structural_features)
        return hashlib.sha256(structural_string.encode('utf-8')).hexdigest()

    def _calculate_tfidf_vector(self, content: str) -> np.ndarray:
        """Calculate TF-IDF vector for content"""
        try:
            # Use existing vectorizer or fit new one
            if hasattr(self.tfidf_vectorizer, 'vocabulary_'):
                vector = self.tfidf_vectorizer.transform([content])
            else:
                # Fit on current content (in practice, should fit on corpus)
                vector = self.tfidf_vectorizer.fit_transform([content])
            
            return vector.toarray()[0]
        except Exception as e:
            logger.warning(f"Error calculating TF-IDF vector: {e}")
            return None

    async def _calculate_image_hash(self, file_path: str) -> str:
        """Calculate perceptual hash for images"""
        try:
            image = Image.open(file_path)
            # Use perceptual hash for similarity detection
            phash = imagehash.phash(image)
            return str(phash)
        except Exception as e:
            logger.warning(f"Error calculating image hash: {e}")
            return None

    def _calculate_hash_similarity(self, hash1: str, hash2: str) -> float:
        """Calculate similarity between two hashes using edit distance"""
        if hash1 == hash2:
            return 1.0
        
        # Use sequence matcher for similarity
        matcher = SequenceMatcher(None, hash1, hash2)
        return matcher.ratio()

    def _calculate_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between vectors"""
        try:
            similarity = cosine_similarity([vec1], [vec2])[0][0]
            return float(similarity)
        except Exception as e:
            logger.warning(f"Error calculating cosine similarity: {e}")
            return 0.0

    def _calculate_image_similarity(self, hash1: str, hash2: str) -> float:
        """Calculate similarity between image hashes"""
        try:
            # Convert hex strings back to imagehash objects
            h1 = imagehash.hex_to_hash(hash1)
            h2 = imagehash.hex_to_hash(hash2)
            
            # Calculate Hamming distance and convert to similarity
            distance = h1 - h2
            max_distance = len(hash1) * 4  # 4 bits per hex character
            similarity = 1.0 - (distance / max_distance)
            
            return max(0.0, similarity)
        except Exception as e:
            logger.warning(f"Error calculating image similarity: {e}")
            return 0.0

    async def _cache_fingerprint(self, fingerprint: DocumentFingerprint):
        """Cache fingerprint to disk"""
        try:
            cache_path = Path(self.config['fingerprint_cache_path'])
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Load existing cache
            cache_data = {}
            if cache_path.exists():
                async with aiofiles.open(cache_path, 'r') as f:
                    content = await f.read()
                    cache_data = json.loads(content)
            
            # Add new fingerprint (convert numpy arrays to lists)
            fp_data = {
                'document_id': fingerprint.document_id,
                'file_hash': fingerprint.file_hash,
                'content_hash': fingerprint.content_hash,
                'structural_hash': fingerprint.structural_hash,
                'tfidf_vector': fingerprint.tfidf_vector.tolist() if fingerprint.tfidf_vector is not None else None,
                'image_hash': fingerprint.image_hash,
                'metadata': fingerprint.metadata
            }
            
            cache_data[fingerprint.document_id] = fp_data
            
            # Save cache
            async with aiofiles.open(cache_path, 'w') as f:
                await f.write(json.dumps(cache_data, indent=2))
                
        except Exception as e:
            logger.warning(f"Error caching fingerprint: {e}")

    async def load_cached_fingerprints(self):
        """Load fingerprints from cache"""
        try:
            cache_path = Path(self.config['fingerprint_cache_path'])
            if not cache_path.exists():
                return
                
            async with aiofiles.open(cache_path, 'r') as f:
                content = await f.read()
                cache_data = json.loads(content)
            
            for doc_id, fp_data in cache_data.items():
                fingerprint = DocumentFingerprint(
                    document_id=fp_data['document_id'],
                    file_hash=fp_data['file_hash'],
                    content_hash=fp_data['content_hash'],
                    structural_hash=fp_data['structural_hash'],
                    tfidf_vector=np.array(fp_data['tfidf_vector']) if fp_data['tfidf_vector'] else None,
                    image_hash=fp_data['image_hash'],
                    metadata=fp_data['metadata']
                )
                self.document_fingerprints[doc_id] = fingerprint
                
            logger.info(f"Loaded {len(cache_data)} fingerprints from cache")
            
        except Exception as e:
            logger.warning(f"Error loading cached fingerprints: {e}")

    async def batch_detect_duplicates(self, document_ids: List[str]) -> Dict[str, List[DuplicateMatch]]:
        """Detect duplicates for multiple documents in batch"""
        results = {}
        
        # Process in batches
        batch_size = self.config['batch_size']
        for i in range(0, len(document_ids), batch_size):
            batch = document_ids[i:i + batch_size]
            
            batch_tasks = []
            for doc_id in batch:
                if doc_id in self.document_fingerprints:
                    task = self.detect_duplicates(doc_id)
                    batch_tasks.append((doc_id, task))
            
            # Execute batch
            for doc_id, task in batch_tasks:
                try:
                    duplicates = await task
                    results[doc_id] = duplicates
                except Exception as e:
                    logger.error(f"Error detecting duplicates for {doc_id}: {e}")
                    results[doc_id] = []
        
        return results

    def get_duplicate_groups(self) -> List[List[str]]:
        """Group documents by their duplicate relationships"""
        # Build graph of duplicate relationships
        duplicate_graph = {}
        
        for match in self.duplicate_matches:
            doc1, doc2 = match.document_id_1, match.document_id_2
            
            if doc1 not in duplicate_graph:
                duplicate_graph[doc1] = set()
            if doc2 not in duplicate_graph:
                duplicate_graph[doc2] = set()
                
            duplicate_graph[doc1].add(doc2)
            duplicate_graph[doc2].add(doc1)
        
        # Find connected components (duplicate groups)
        visited = set()
        groups = []
        
        def dfs(node, current_group):
            if node in visited:
                return
            visited.add(node)
            current_group.append(node)
            
            for neighbor in duplicate_graph.get(node, []):
                dfs(neighbor, current_group)
        
        for doc_id in duplicate_graph:
            if doc_id not in visited:
                group = []
                dfs(doc_id, group)
                if len(group) > 1:  # Only groups with actual duplicates
                    groups.append(group)
        
        return groups

    def get_statistics(self) -> Dict:
        """Get duplicate detection statistics"""
        total_documents = len(self.document_fingerprints)
        total_matches = len(self.duplicate_matches)
        
        # Count by duplicate type
        type_counts = {}
        for match in self.duplicate_matches:
            type_name = match.duplicate_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        # Calculate duplicate groups
        groups = self.get_duplicate_groups()
        documents_with_duplicates = sum(len(group) for group in groups)
        
        return {
            'total_documents': total_documents,
            'total_duplicate_matches': total_matches,
            'duplicate_groups': len(groups),
            'documents_with_duplicates': documents_with_duplicates,
            'duplicate_rate': documents_with_duplicates / total_documents if total_documents > 0 else 0,
            'matches_by_type': type_counts,
            'average_confidence': sum(match.confidence for match in self.duplicate_matches) / total_matches if total_matches > 0 else 0
        }


# Factory function for easy instantiation
def create_duplicate_detector(config: Dict = None) -> DuplicateDetector:
    """Create a duplicate detector instance with configuration"""
    return DuplicateDetector(config)