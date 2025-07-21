"""
Confluence Data Source Plugin
Fetches documents from Atlassian Confluence.
"""

import aiohttp
import asyncio
from datetime import datetime
from typing import List, Dict
from plugins.plugin_system import DataSourcePlugin, PluginMetadata, PluginType


class ConfluencePlugin(DataSourcePlugin):
    """Plugin for fetching documents from Confluence"""
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="confluence_source",
            version="1.0.0",
            description="Fetches documents from Atlassian Confluence",
            author="RAG System",
            plugin_type=PluginType.DATA_SOURCE,
            dependencies=[],
            tags=["confluence", "atlassian", "wiki", "collaboration"]
        )
    
    async def initialize(self) -> bool:
        """Initialize the Confluence plugin"""
        required_config = ['base_url', 'username', 'api_token']
        
        for key in required_config:
            if key not in self.config:
                self.logger.error(f"Missing required configuration: {key}")
                return False
        
        self.base_url = self.config['base_url'].rstrip('/')
        self.auth = aiohttp.BasicAuth(self.config['username'], self.config['api_token'])
        self.space_keys = self.config.get('space_keys', [])
        self.page_limit = self.config.get('page_limit', 100)
        
        # Test connection
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/rest/api/content",
                    auth=self.auth,
                    params={'limit': 1}
                ) as response:
                    if response.status == 200:
                        self.logger.info("Successfully connected to Confluence")
                        return True
                    else:
                        self.logger.error(f"Failed to connect to Confluence: {response.status}")
                        return False
        except Exception as e:
            self.logger.error(f"Error connecting to Confluence: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup plugin resources"""
        pass
    
    def get_config_schema(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "base_url": {
                    "type": "string",
                    "description": "Confluence base URL (e.g., https://company.atlassian.net/wiki)"
                },
                "username": {
                    "type": "string", 
                    "description": "Confluence username/email"
                },
                "api_token": {
                    "type": "string",
                    "description": "Confluence API token"
                },
                "space_keys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of space keys to fetch from (empty for all spaces)"
                },
                "page_limit": {
                    "type": "integer",
                    "default": 100,
                    "description": "Maximum number of pages to fetch per request"
                }
            },
            "required": ["base_url", "username", "api_token"]
        }
    
    async def fetch_documents(self, source_config: Dict) -> List[Dict]:
        """Fetch documents from Confluence"""
        documents = []
        
        try:
            async with aiohttp.ClientSession() as session:
                if self.space_keys:
                    # Fetch from specific spaces
                    for space_key in self.space_keys:
                        space_docs = await self._fetch_space_pages(session, space_key)
                        documents.extend(space_docs)
                else:
                    # Fetch from all accessible spaces
                    spaces = await self._fetch_spaces(session)
                    for space in spaces:
                        space_docs = await self._fetch_space_pages(session, space['key'])
                        documents.extend(space_docs)
            
            self.logger.info(f"Fetched {len(documents)} documents from Confluence")
            return documents
            
        except Exception as e:
            self.logger.error(f"Error fetching Confluence documents: {e}")
            return []
    
    async def sync_documents(self, last_sync: datetime) -> List[Dict]:
        """Sync documents modified since last sync"""
        documents = []
        
        try:
            # Convert datetime to Confluence CQL format
            sync_date = last_sync.strftime('%Y-%m-%d')
            
            async with aiohttp.ClientSession() as session:
                # Use CQL to find recently modified pages
                cql_query = f"lastModified >= '{sync_date}'"
                if self.space_keys:
                    space_filter = " OR ".join([f"space = {key}" for key in self.space_keys])
                    cql_query += f" AND ({space_filter})"
                
                params = {
                    'cql': cql_query,
                    'limit': self.page_limit,
                    'expand': 'body.storage,metadata.labels,space,version,ancestors'
                }
                
                async with session.get(
                    f"{self.base_url}/rest/api/content/search",
                    auth=self.auth,
                    params=params
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        for page in data.get('results', []):
                            doc = await self._convert_page_to_document(page)
                            if doc:
                                documents.append(doc)
            
            self.logger.info(f"Synced {len(documents)} modified documents from Confluence")
            return documents
            
        except Exception as e:
            self.logger.error(f"Error syncing Confluence documents: {e}")
            return []
    
    async def _fetch_spaces(self, session: aiohttp.ClientSession) -> List[Dict]:
        """Fetch all accessible spaces"""
        spaces = []
        start = 0
        
        while True:
            params = {
                'start': start,
                'limit': self.page_limit
            }
            
            async with session.get(
                f"{self.base_url}/rest/api/space",
                auth=self.auth,
                params=params
            ) as response:
                if response.status != 200:
                    break
                
                data = await response.json()
                batch = data.get('results', [])
                spaces.extend(batch)
                
                if len(batch) < self.page_limit:
                    break
                
                start += self.page_limit
        
        return spaces
    
    async def _fetch_space_pages(self, session: aiohttp.ClientSession, space_key: str) -> List[Dict]:
        """Fetch all pages from a specific space"""
        documents = []
        start = 0
        
        while True:
            params = {
                'spaceKey': space_key,
                'start': start,
                'limit': self.page_limit,
                'expand': 'body.storage,metadata.labels,space,version,ancestors'
            }
            
            async with session.get(
                f"{self.base_url}/rest/api/content",
                auth=self.auth,
                params=params
            ) as response:
                if response.status != 200:
                    self.logger.warning(f"Failed to fetch pages from space {space_key}: {response.status}")
                    break
                
                data = await response.json()
                pages = data.get('results', [])
                
                for page in pages:
                    doc = await self._convert_page_to_document(page)
                    if doc:
                        documents.append(doc)
                
                if len(pages) < self.page_limit:
                    break
                
                start += self.page_limit
        
        return documents
    
    async def _convert_page_to_document(self, page: Dict) -> Dict:
        """Convert Confluence page to document format"""
        try:
            # Extract content
            body = page.get('body', {}).get('storage', {}).get('value', '')
            
            # Clean HTML content (basic cleaning)
            import re
            content = re.sub(r'<[^>]+>', '', body)  # Remove HTML tags
            content = re.sub(r'\s+', ' ', content).strip()  # Normalize whitespace
            
            # Extract metadata
            labels = [label['name'] for label in page.get('metadata', {}).get('labels', {}).get('results', [])]
            
            # Build ancestor path
            ancestors = page.get('ancestors', [])
            path_parts = [ancestor['title'] for ancestor in ancestors]
            path_parts.append(page['title'])
            document_path = ' > '.join(path_parts)
            
            document = {
                'source_id': page['id'],
                'title': page['title'],
                'content': content,
                'url': f"{self.base_url}/spaces/{page['space']['key']}/pages/{page['id']}",
                'metadata': {
                    'source': 'confluence',
                    'space_key': page['space']['key'],
                    'space_name': page['space']['name'],
                    'page_id': page['id'],
                    'version': page['version']['number'],
                    'labels': labels,
                    'path': document_path,
                    'created_date': page['version']['when'],
                    'modified_date': page['version']['when'],
                    'author': page['version']['by']['displayName']
                },
                'file_type': 'html',
                'source_type': 'confluence'
            }
            
            return document
            
        except Exception as e:
            self.logger.error(f"Error converting Confluence page to document: {e}")
            return None