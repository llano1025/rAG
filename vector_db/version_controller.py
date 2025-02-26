from typing import Dict, List, Optional
import datetime
from dataclasses import dataclass
from .chunking import Chunk

class VersionError(Exception):
    """Raised when version operation fails."""
    pass

@dataclass
class Version:
    """Represents a version of a document."""
    id: str
    document_id: str
    timestamp: datetime.datetime
    chunks: List[Chunk]
    metadata: Dict
    parent_version: Optional[str] = None

class VersionController:
    """Manages document versions and their associated embeddings."""
    
    def __init__(self):
        self.versions: Dict[str, Version] = {}
        self.current_versions: Dict[str, str] = {}  # document_id -> version_id

    def create_version(
        self,
        document_id: str,
        chunks: List[Chunk],
        metadata: Dict,
        parent_version: Optional[str] = None
    ) -> Version:
        """Create a new version for a document."""
        try:
            version_id = self._generate_version_id(document_id)
            
            version = Version(
                id=version_id,
                document_id=document_id,
                timestamp=datetime.datetime.now(),
                chunks=chunks,
                metadata=metadata,
                parent_version=parent_version
            )
            
            self.versions[version_id] = version
            self.current_versions[document_id] = version_id
            
            return version
        except Exception as e:
            raise VersionError(f"Failed to create version: {str(e)}")

    def get_version(self, version_id: str) -> Optional[Version]:
        """Retrieve a specific version."""
        return self.versions.get(version_id)

    def get_current_version(self, document_id: str) -> Optional[Version]:
        """Get the current version of a document."""
        version_id = self.current_versions.get(document_id)
        return self.versions.get(version_id) if version_id else None

    def get_version_history(self, document_id: str) -> List[Version]:
        """Get the version history of a document."""
        try:
            versions = []
            current_version_id = self.current_versions.get(document_id)
            
            while current_version_id:
                version = self.versions.get(current_version_id)
                if version:
                    versions.append(version)
                    current_version_id = version.parent_version
                else:
                    break
            
            return versions
        except Exception as e:
            raise VersionError(f"Failed to retrieve version history: {str(e)}")

    def _generate_version_id(self, document_id: str) -> str:
        """Generate a unique version ID."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        return f"{document_id}_v{timestamp}"