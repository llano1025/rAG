from typing import Dict, List, Optional, Any
import aiohttp
import asyncio
import json
import logging
from datetime import datetime
from pydantic import BaseModel, HttpUrl
from enum import Enum
import hmac
import hashlib
import time
from fastapi import HTTPException

logger = logging.getLogger(__name__)

class WebhookEvent(str, Enum):
    DOCUMENT_CREATED = "document.created"
    DOCUMENT_UPDATED = "document.updated"
    DOCUMENT_DELETED = "document.deleted"
    EXPORT_COMPLETED = "export.completed"
    VECTOR_UPDATED = "vector.updated"
    ERROR_OCCURRED = "error.occurred"

class WebhookConfig(BaseModel):
    url: HttpUrl
    secret: str
    events: List[WebhookEvent]
    is_active: bool = True
    retry_count: int = 3
    timeout: int = 10

class WebhookDelivery(BaseModel):
    id: str
    webhook_id: str
    event: WebhookEvent
    payload: Dict
    timestamp: datetime
    status: str
    response_status: Optional[int] = None
    response_body: Optional[str] = None
    retry_count: int = 0

class WebhookManager:
    """
    Manages webhook configurations, deliveries, and retry logic
    """
    
    def __init__(self, storage_client: Any):
        self.storage = storage_client
        self.webhook_configs: Dict[str, WebhookConfig] = {}
        self.max_payload_size = 5 * 1024 * 1024  # 5MB
        
    async def register_webhook(self, config: WebhookConfig) -> str:
        """
        Register a new webhook configuration
        
        Args:
            config: WebhookConfig instance
            
        Returns:
            str: Webhook ID
        """
        webhook_id = self._generate_webhook_id()
        self.webhook_configs[webhook_id] = config
        await self._save_webhook_config(webhook_id, config)
        return webhook_id
        
    async def trigger_event(
        self,
        event: WebhookEvent,
        payload: Dict,
        immediate: bool = True
    ) -> List[str]:
        """
        Trigger webhook event and deliver to registered webhooks
        
        Args:
            event: Type of webhook event
            payload: Event payload
            immediate: Whether to deliver immediately or queue
            
        Returns:
            List[str]: List of delivery IDs
        """
        if len(json.dumps(payload)) > self.max_payload_size:
            raise ValueError("Payload size exceeds maximum limit")
            
        delivery_ids = []
        
        for webhook_id, config in self.webhook_configs.items():
            if not config.is_active or event not in config.events:
                continue
                
            delivery_id = await self._create_delivery(webhook_id, event, payload)
            delivery_ids.append(delivery_id)
            
            if immediate:
                asyncio.create_task(self._deliver_webhook(delivery_id))
            else:
                await self._queue_delivery(delivery_id)
                
        return delivery_ids

    async def _deliver_webhook(self, delivery_id: str) -> None:
        """
        Deliver webhook with retry logic
        
        Args:
            delivery_id: ID of webhook delivery to process
        """
        delivery = await self._get_delivery(delivery_id)
        config = self.webhook_configs[delivery.webhook_id]
        
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "RAG-System-Webhook/1.0",
            "X-Webhook-ID": delivery.webhook_id,
            "X-Delivery-ID": delivery_id,
            "X-Event-Type": delivery.event.value,
            "X-Timestamp": str(int(time.time())),
        }
        
        # Add signature for security
        payload_str = json.dumps(delivery.payload)
        signature = self._generate_signature(payload_str, config.secret)
        headers["X-Signature"] = signature
        
        async with aiohttp.ClientSession() as session:
            for attempt in range(config.retry_count + 1):
                try:
                    async with session.post(
                        str(config.url),
                        json=delivery.payload,
                        headers=headers,
                        timeout=config.timeout
                    ) as response:
                        response_body = await response.text()
                        
                        await self._update_delivery_status(
                            delivery_id,
                            "completed" if response.status < 400 else "failed",
                            response.status,
                            response_body
                        )
                        
                        if response.status < 400:
                            logger.info(f"Webhook delivered successfully: {delivery_id}")
                            return
                            
                        logger.warning(
                            f"Webhook delivery failed with status {response.status}: {delivery_id}"
                        )
                        
                except asyncio.TimeoutError:
                    logger.warning(f"Webhook delivery timeout: {delivery_id}")
                except Exception as e:
                    logger.error(f"Webhook delivery error: {str(e)}")
                    
                if attempt < config.retry_count:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    
            await self._update_delivery_status(delivery_id, "failed")
            logger.error(f"Webhook delivery failed after {config.retry_count} retries: {delivery_id}")

    def _generate_signature(self, payload: str, secret: str) -> str:
        """Generate HMAC signature for webhook payload"""
        return hmac.new(
            secret.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()

    async def get_webhook_deliveries(
        self,
        webhook_id: str,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[WebhookDelivery]:
        """
        Get webhook delivery history
        
        Args:
            webhook_id: ID of webhook
            status: Filter by delivery status
            limit: Maximum number of deliveries to return
            
        Returns:
            List[WebhookDelivery]: List of webhook deliveries
        """
        # Implementation for retrieving delivery history
        pass

    async def retry_delivery(self, delivery_id: str) -> None:
        """
        Manually retry a failed webhook delivery
        
        Args:
            delivery_id: ID of delivery to retry
        """
        delivery = await self._get_delivery(delivery_id)
        if delivery.status != "failed":
            raise ValueError("Can only retry failed deliveries")
            
        delivery.retry_count = 0
        await self._update_delivery(delivery)
        await self._deliver_webhook(delivery_id)

    async def _save_webhook_config(self, webhook_id: str, config: WebhookConfig) -> None:
        """Save webhook configuration to storage"""
        await self.storage.save_webhook_config(webhook_id, config.dict())

    async def _get_delivery(self, delivery_id: str) -> WebhookDelivery:
        """Get webhook delivery from storage"""
        delivery_data = await self.storage.get_webhook_delivery(delivery_id)
        if not delivery_data:
            raise HTTPException(status_code=404, detail="Webhook delivery not found")
        return WebhookDelivery(**delivery_data)

    async def _update_delivery_status(
        self,
        delivery_id: str,
        status: str,
        response_status: Optional[int] = None,
        response_body: Optional[str] = None
    ) -> None:
        """Update webhook delivery status in storage"""
        delivery = await self._get_delivery(delivery_id)
        delivery.status = status
        delivery.response_status = response_status
        delivery.response_body = response_body
        await self._update_delivery(delivery)

    async def _update_delivery(self, delivery: WebhookDelivery) -> None:
        """Update webhook delivery in storage"""
        await self.storage.update_webhook_delivery(delivery.id, delivery.dict())

    def _generate_webhook_id(self) -> str:
        """Generate unique webhook ID"""
        return f"whk_{int(time.time())}_{hash(str(time.time_ns()))}"

    async def _queue_delivery(self, delivery_id: str) -> None:
        """Queue webhook delivery for processing"""
        # Implementation for queuing delivery (e.g., using Redis)
        pass