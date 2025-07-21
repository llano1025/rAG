"""
Google Drive Data Source Plugin
Fetches documents from Google Drive.
"""

import aiohttp
import asyncio
from datetime import datetime
from typing import List, Dict
import json
import base64
from plugins.plugin_system import DataSourcePlugin, PluginMetadata, PluginType


class GoogleDrivePlugin(DataSourcePlugin):
    """Plugin for fetching documents from Google Drive"""
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="google_drive_source",
            version="1.0.0", 
            description="Fetches documents from Google Drive",
            author="RAG System",
            plugin_type=PluginType.DATA_SOURCE,
            dependencies=[],
            tags=["google", "drive", "cloud", "documents"]
        )
    
    async def initialize(self) -> bool:
        """Initialize the Google Drive plugin"""
        required_config = ['service_account_key']
        
        for key in required_config:
            if key not in self.config:
                self.logger.error(f"Missing required configuration: {key}")
                return False
        
        # Load service account credentials
        try:
            if isinstance(self.config['service_account_key'], str):
                # If it's a file path
                with open(self.config['service_account_key'], 'r') as f:
                    self.service_account = json.load(f)
            else:
                # If it's the JSON directly
                self.service_account = self.config['service_account_key']
            
            self.folder_ids = self.config.get('folder_ids', [])
            self.file_types = self.config.get('file_types', ['application/pdf', 'application/vnd.google-apps.document'])
            self.max_files = self.config.get('max_files', 1000)
            
            # Get access token
            self.access_token = await self._get_access_token()
            if self.access_token:
                self.logger.info("Successfully authenticated with Google Drive")
                return True
            else:
                self.logger.error("Failed to authenticate with Google Drive")
                return False
                
        except Exception as e:
            self.logger.error(f"Error initializing Google Drive plugin: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup plugin resources"""
        pass
    
    def get_config_schema(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "service_account_key": {
                    "description": "Path to service account JSON file or the JSON content",
                    "oneOf": [
                        {"type": "string"},
                        {"type": "object"}
                    ]
                },
                "folder_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of Google Drive folder IDs to fetch from (empty for all accessible files)"
                },
                "file_types": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "default": ["application/pdf", "application/vnd.google-apps.document"],
                    "description": "MIME types of files to fetch"
                },
                "max_files": {
                    "type": "integer",
                    "default": 1000,
                    "description": "Maximum number of files to fetch"
                }
            },
            "required": ["service_account_key"]
        }
    
    async def _get_access_token(self) -> str:
        """Get OAuth2 access token using service account"""
        try:
            import jwt
            import time
            
            # Create JWT
            now = int(time.time())
            payload = {
                'iss': self.service_account['client_email'],
                'scope': 'https://www.googleapis.com/auth/drive.readonly',
                'aud': 'https://oauth2.googleapis.com/token',
                'exp': now + 3600,
                'iat': now
            }
            
            # Sign JWT
            token = jwt.encode(payload, self.service_account['private_key'], algorithm='RS256')
            
            # Exchange JWT for access token
            async with aiohttp.ClientSession() as session:
                data = {
                    'grant_type': 'urn:ietf:params:oauth:grant-type:jwt-bearer',
                    'assertion': token
                }
                
                async with session.post(
                    'https://oauth2.googleapis.com/token',
                    data=data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result['access_token']
                    else:
                        self.logger.error(f"Failed to get access token: {response.status}")
                        return None
                        
        except Exception as e:
            self.logger.error(f"Error getting access token: {e}")
            return None
    
    async def fetch_documents(self, source_config: Dict) -> List[Dict]:
        """Fetch documents from Google Drive"""
        documents = []
        
        try:
            headers = {'Authorization': f'Bearer {self.access_token}'}
            
            async with aiohttp.ClientSession() as session:
                if self.folder_ids:
                    # Fetch from specific folders
                    for folder_id in self.folder_ids:
                        folder_docs = await self._fetch_folder_files(session, folder_id, headers)
                        documents.extend(folder_docs)
                else:
                    # Fetch all accessible files
                    all_docs = await self._fetch_all_files(session, headers)
                    documents.extend(all_docs)
            
            self.logger.info(f"Fetched {len(documents)} documents from Google Drive")
            return documents
            
        except Exception as e:
            self.logger.error(f"Error fetching Google Drive documents: {e}")
            return []
    
    async def sync_documents(self, last_sync: datetime) -> List[Dict]:
        """Sync documents modified since last sync"""
        documents = []
        
        try:
            headers = {'Authorization': f'Bearer {self.access_token}'}
            
            # Convert datetime to RFC 3339 format
            sync_time = last_sync.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            
            # Build query for modified files
            query_parts = [f"modifiedTime > '{sync_time}'"]
            
            if self.file_types:
                mime_query = " or ".join([f"mimeType = '{mt}'" for mt in self.file_types])
                query_parts.append(f"({mime_query})")
            
            if self.folder_ids:
                folder_query = " or ".join([f"'{fid}' in parents" for fid in self.folder_ids])
                query_parts.append(f"({folder_query})")
            
            query = " and ".join(query_parts)
            
            async with aiohttp.ClientSession() as session:
                params = {
                    'q': query,
                    'fields': 'files(id,name,mimeType,modifiedTime,createdTime,size,webViewLink,parents,owners)',
                    'pageSize': min(self.max_files, 1000)
                }
                
                async with session.get(
                    'https://www.googleapis.com/drive/v3/files',
                    headers=headers,
                    params=params
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        for file_info in data.get('files', []):
                            doc = await self._convert_file_to_document(session, file_info, headers)
                            if doc:
                                documents.append(doc)
            
            self.logger.info(f"Synced {len(documents)} modified documents from Google Drive")
            return documents
            
        except Exception as e:
            self.logger.error(f"Error syncing Google Drive documents: {e}")
            return []
    
    async def _fetch_all_files(self, session: aiohttp.ClientSession, headers: Dict) -> List[Dict]:
        """Fetch all accessible files"""
        documents = []
        next_page_token = None
        
        while len(documents) < self.max_files:
            # Build query
            query_parts = []
            if self.file_types:
                mime_query = " or ".join([f"mimeType = '{mt}'" for mt in self.file_types])
                query_parts.append(f"({mime_query})")
            
            query = " and ".join(query_parts) if query_parts else None
            
            params = {
                'fields': 'files(id,name,mimeType,modifiedTime,createdTime,size,webViewLink,parents,owners),nextPageToken',
                'pageSize': min(1000, self.max_files - len(documents))
            }
            
            if query:
                params['q'] = query
            if next_page_token:
                params['pageToken'] = next_page_token
            
            async with session.get(
                'https://www.googleapis.com/drive/v3/files',
                headers=headers,
                params=params
            ) as response:
                if response.status != 200:
                    break
                
                data = await response.json()
                files = data.get('files', [])
                
                for file_info in files:
                    doc = await self._convert_file_to_document(session, file_info, headers)
                    if doc:
                        documents.append(doc)
                
                next_page_token = data.get('nextPageToken')
                if not next_page_token or len(files) == 0:
                    break
        
        return documents
    
    async def _fetch_folder_files(self, session: aiohttp.ClientSession, folder_id: str, headers: Dict) -> List[Dict]:
        """Fetch files from a specific folder"""
        documents = []
        next_page_token = None
        
        while len(documents) < self.max_files:
            # Build query for folder
            query_parts = [f"'{folder_id}' in parents"]
            if self.file_types:
                mime_query = " or ".join([f"mimeType = '{mt}'" for mt in self.file_types])
                query_parts.append(f"({mime_query})")
            
            query = " and ".join(query_parts)
            
            params = {
                'q': query,
                'fields': 'files(id,name,mimeType,modifiedTime,createdTime,size,webViewLink,parents,owners),nextPageToken',
                'pageSize': min(1000, self.max_files - len(documents))
            }
            
            if next_page_token:
                params['pageToken'] = next_page_token
            
            async with session.get(
                'https://www.googleapis.com/drive/v3/files',
                headers=headers,
                params=params
            ) as response:
                if response.status != 200:
                    break
                
                data = await response.json()
                files = data.get('files', [])
                
                for file_info in files:
                    doc = await self._convert_file_to_document(session, file_info, headers)
                    if doc:
                        documents.append(doc)
                
                next_page_token = data.get('nextPageToken')
                if not next_page_token or len(files) == 0:
                    break
        
        return documents
    
    async def _convert_file_to_document(self, session: aiohttp.ClientSession, file_info: Dict, headers: Dict) -> Dict:
        """Convert Google Drive file to document format"""
        try:
            # Get file content
            content = ""
            
            if file_info['mimeType'] == 'application/vnd.google-apps.document':
                # Export Google Doc as plain text
                export_url = f"https://www.googleapis.com/drive/v3/files/{file_info['id']}/export"
                params = {'mimeType': 'text/plain'}
                
                async with session.get(export_url, headers=headers, params=params) as response:
                    if response.status == 200:
                        content = await response.text()
            
            elif file_info['mimeType'] == 'application/pdf':
                # For PDFs, we'd need to download and extract text
                # For now, just use metadata
                content = f"PDF Document: {file_info['name']}"
            
            # Get folder path
            folder_path = await self._get_folder_path(session, file_info.get('parents', []), headers)
            
            document = {
                'source_id': file_info['id'],
                'title': file_info['name'],
                'content': content,
                'url': file_info.get('webViewLink', ''),
                'metadata': {
                    'source': 'google_drive',
                    'file_id': file_info['id'],
                    'mime_type': file_info['mimeType'],
                    'size': file_info.get('size', 0),
                    'folder_path': folder_path,
                    'created_date': file_info.get('createdTime'),
                    'modified_date': file_info.get('modifiedTime'),
                    'owners': [owner.get('displayName', owner.get('emailAddress', '')) for owner in file_info.get('owners', [])]
                },
                'file_type': self._get_file_extension(file_info['mimeType']),
                'source_type': 'google_drive'
            }
            
            return document
            
        except Exception as e:
            self.logger.error(f"Error converting Google Drive file to document: {e}")
            return None
    
    async def _get_folder_path(self, session: aiohttp.ClientSession, parent_ids: List[str], headers: Dict) -> str:
        """Get the full folder path for a file"""
        if not parent_ids:
            return "/"
        
        try:
            # Get parent folder info
            parent_id = parent_ids[0]  # Take first parent
            
            async with session.get(
                f'https://www.googleapis.com/drive/v3/files/{parent_id}',
                headers=headers,
                params={'fields': 'name,parents'}
            ) as response:
                if response.status == 200:
                    folder_info = await response.json()
                    folder_name = folder_info.get('name', 'Unknown')
                    
                    # Recursively get parent path
                    if folder_info.get('parents'):
                        parent_path = await self._get_folder_path(session, folder_info['parents'], headers)
                        return f"{parent_path}/{folder_name}"
                    else:
                        return f"/{folder_name}"
                        
        except Exception as e:
            self.logger.warning(f"Error getting folder path: {e}")
        
        return "/Unknown"
    
    def _get_file_extension(self, mime_type: str) -> str:
        """Get file extension from MIME type"""
        mime_to_ext = {
            'application/pdf': 'pdf',
            'application/vnd.google-apps.document': 'gdoc',
            'application/vnd.google-apps.spreadsheet': 'gsheet',
            'application/vnd.google-apps.presentation': 'gslides',
            'text/plain': 'txt',
            'application/msword': 'doc',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx'
        }
        
        return mime_to_ext.get(mime_type, 'unknown')