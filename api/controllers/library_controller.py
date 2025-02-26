# api/controllers/library_controller.py
from typing import List, Optional
from datetime import datetime
from fastapi import HTTPException

from utils.security.encryption import EncryptionManager
from utils.security.audit_logger import AuditLogger
from utils.security.pii_detector import PIIDetector

class LibraryController:
    def __init__(
        self,
        encryption: EncryptionManager,
        audit_logger: AuditLogger,
        pii_detector: PIIDetector
    ):
        self.encryption = encryption
        self.audit_logger = audit_logger
        self.pii_detector = pii_detector

    async def manage_folder(
        self,
        operation: str,
        folder_data: dict,
        user_id: str
    ) -> dict:
        try:
            # Check for PII in folder data
            pii_check = self.pii_detector.detect_pii(str(folder_data))
            if pii_check:
                folder_data = self.pii_detector.redact_pii(str(folder_data))
            
            # Encrypt sensitive data
            encrypted_data = self.encryption.encrypt_data(folder_data)
            
            # Perform operation
            result = await self._execute_folder_operation(operation, encrypted_data)
            
            # Audit logging
            await self.audit_logger.log(
                action=operation,
                resource_type="folder",
                user_id=user_id,
                resource_id=result.get("id"),
                details={"pii_detected": bool(pii_check)}
            )
            
            return result
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))