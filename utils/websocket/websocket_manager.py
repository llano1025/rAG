"""
WebSocket manager for real-time progress updates and notifications.
Provides progress tracking for document processing, embedding generation, and other long-running tasks.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
import socketio
from fastapi import FastAPI

logger = logging.getLogger(__name__)

class WebSocketManager:
    """Manages WebSocket connections and real-time progress broadcasting."""

    def __init__(self):
        """Initialize WebSocket manager with Socket.IO server."""
        self.sio = socketio.AsyncServer(
            cors_allowed_origins="*",
            async_mode='asgi',
            logger=False,
            engineio_logger=False
        )
        self.connected_users: Dict[str, str] = {}  # socket_id -> user_id mapping
        self.user_sockets: Dict[str, List[str]] = {}  # user_id -> list of socket_ids

        # Setup event handlers
        self._setup_handlers()

    def _setup_handlers(self):
        """Setup WebSocket event handlers."""

        @self.sio.event
        async def connect(sid, environ, auth):
            """Handle client connection."""
            try:
                # Extract auth token from connection
                token = None
                if auth and 'token' in auth:
                    token = auth['token']
                elif 'HTTP_AUTHORIZATION' in environ:
                    auth_header = environ['HTTP_AUTHORIZATION']
                    if auth_header.startswith('Bearer '):
                        token = auth_header[7:]

                if not token:
                    logger.warning(f"WebSocket connection {sid} rejected: No auth token")
                    await self.sio.disconnect(sid)
                    return False

                # Validate token and get user ID (simplified - in real implementation, verify JWT)
                # For now, we'll assume token validation is handled elsewhere
                user_id = self._extract_user_id_from_token(token)
                if not user_id:
                    logger.warning(f"WebSocket connection {sid} rejected: Invalid token")
                    await self.sio.disconnect(sid)
                    return False

                # Track connection
                self.connected_users[sid] = user_id
                if user_id not in self.user_sockets:
                    self.user_sockets[user_id] = []
                self.user_sockets[user_id].append(sid)

                logger.info(f"WebSocket client {sid} connected for user {user_id}")

                # Send connection confirmation
                await self.sio.emit('connected', {'status': 'connected', 'user_id': user_id}, room=sid)
                return True

            except Exception as e:
                logger.error(f"Error handling WebSocket connection {sid}: {e}")
                await self.sio.disconnect(sid)
                return False

        @self.sio.event
        async def disconnect(sid):
            """Handle client disconnection."""
            try:
                user_id = self.connected_users.get(sid)
                if user_id:
                    # Remove from tracking
                    if sid in self.connected_users:
                        del self.connected_users[sid]

                    if user_id in self.user_sockets:
                        self.user_sockets[user_id] = [s for s in self.user_sockets[user_id] if s != sid]
                        if not self.user_sockets[user_id]:
                            del self.user_sockets[user_id]

                    logger.info(f"WebSocket client {sid} disconnected for user {user_id}")
                else:
                    logger.info(f"WebSocket client {sid} disconnected (no user tracked)")

            except Exception as e:
                logger.error(f"Error handling WebSocket disconnection {sid}: {e}")

    def _extract_user_id_from_token(self, token: str) -> Optional[str]:
        """Extract user ID from JWT token (simplified implementation)."""
        try:
            # In a real implementation, you would decode and validate the JWT
            # For now, return a placeholder to enable functionality
            # This should be replaced with proper JWT validation
            import base64
            import json

            # Try to decode JWT payload (this is simplified and not secure)
            try:
                # Split JWT token
                parts = token.split('.')
                if len(parts) >= 2:
                    # Decode payload (add padding if needed)
                    payload = parts[1]
                    payload += '=' * (4 - len(payload) % 4)  # Add padding
                    decoded = base64.urlsafe_b64decode(payload)
                    data = json.loads(decoded)
                    return str(data.get('sub', data.get('user_id', 'unknown')))
            except:
                pass

            # Fallback: return a hash of the token as user identifier
            return str(hash(token) % 10000)

        except Exception as e:
            logger.error(f"Error extracting user ID from token: {e}")
            return None

    async def emit_to_user(self, user_id: str, event: str, data: Any):
        """Emit event to all sockets for a specific user."""
        try:
            if user_id in self.user_sockets:
                for socket_id in self.user_sockets[user_id]:
                    await self.sio.emit(event, data, room=socket_id)
                logger.debug(f"Emitted {event} to user {user_id} ({len(self.user_sockets[user_id])} sockets)")
            else:
                logger.debug(f"No active sockets for user {user_id}")
        except Exception as e:
            logger.error(f"Error emitting to user {user_id}: {e}")

    async def emit_document_progress(
        self,
        user_id: str,
        document_id: int,
        filename: str,
        stage: str,
        progress: int,
        estimated_remaining_seconds: Optional[int] = None,
        chunks_processed: Optional[int] = None,
        total_chunks: Optional[int] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ):
        """Emit document processing progress update."""
        data = {
            'type': 'document_processing_progress',
            'document_id': document_id,
            'filename': filename,
            'stage': stage,
            'progress': progress,
            'timestamp': datetime.utcnow().isoformat(),
        }

        if estimated_remaining_seconds is not None:
            data['estimated_remaining_seconds'] = estimated_remaining_seconds

        if chunks_processed is not None and total_chunks is not None:
            data['chunks_processed'] = chunks_processed
            data['total_chunks'] = total_chunks

        if additional_data:
            data.update(additional_data)

        await self.emit_to_user(user_id, 'document_processing_progress', data)

    async def emit_document_complete(self, user_id: str, document_id: int, filename: str, success: bool, error_message: Optional[str] = None):
        """Emit document processing completion."""
        data = {
            'type': 'document_processing_complete',
            'document_id': document_id,
            'filename': filename,
            'success': success,
            'timestamp': datetime.utcnow().isoformat(),
        }

        if error_message:
            data['error_message'] = error_message

        event = 'document_processed' if success else 'document_failed'
        await self.emit_to_user(user_id, event, data)

    async def emit_system_alert(self, user_id: str, level: str, message: str, details: Optional[Dict[str, Any]] = None):
        """Emit system alert to user."""
        data = {
            'type': 'system_alert',
            'level': level,  # info, warning, error
            'message': message,
            'timestamp': datetime.utcnow().isoformat(),
        }

        if details:
            data['details'] = details

        await self.emit_to_user(user_id, 'system_alert', data)

    def mount_to_app(self, app: FastAPI, path: str = "/ws"):
        """Mount WebSocket server to FastAPI app."""
        socket_app = socketio.ASGIApp(self.sio, other_asgi_app=app)
        app.mount(path, socket_app)
        logger.info(f"WebSocket server mounted at {path}")

    def get_connected_users(self) -> List[str]:
        """Get list of connected user IDs."""
        return list(self.user_sockets.keys())

    def get_user_socket_count(self, user_id: str) -> int:
        """Get number of active sockets for a user."""
        return len(self.user_sockets.get(user_id, []))

# Global WebSocket manager instance
_websocket_manager: Optional[WebSocketManager] = None

def get_websocket_manager() -> WebSocketManager:
    """Get global WebSocket manager instance."""
    global _websocket_manager
    if _websocket_manager is None:
        _websocket_manager = WebSocketManager()
    return _websocket_manager

def init_websocket_manager(app: FastAPI):
    """Initialize and mount WebSocket manager to FastAPI app."""
    manager = get_websocket_manager()
    manager.mount_to_app(app)
    return manager