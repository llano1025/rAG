from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import os
import base64
from typing import Union, Optional
import logging
from dataclasses import dataclass
from pathlib import Path

@dataclass
class EncryptionConfig:
    key_path: Path
    salt_path: Path
    iterations: int = 100_000

class EncryptionManager:
    """Handles encryption and decryption of sensitive data using Fernet (symmetric encryption)."""
    
    def __init__(self, config: EncryptionConfig):
        self.config = config
        self._fernet = None
        self._initialize_encryption()

    def _initialize_encryption(self) -> None:
        """Initialize the encryption system, creating or loading necessary keys."""
        try:
            if not self.config.key_path.exists():
                self._generate_new_key()
            else:
                self._load_existing_key()
                
            if not self.config.salt_path.exists():
                self._generate_new_salt()
            else:
                self._load_existing_salt()
                
        except Exception as e:
            logging.error(f"Failed to initialize encryption: {str(e)}")
            raise RuntimeError("Encryption initialization failed")

    def _generate_new_key(self) -> None:
        """Generate a new encryption key and save it securely."""
        key = Fernet.generate_key()
        self.config.key_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.key_path.write_bytes(key)
        self._fernet = Fernet(key)

    def _load_existing_key(self) -> None:
        """Load an existing encryption key."""
        key = self.config.key_path.read_bytes()
        self._fernet = Fernet(key)

    def _generate_new_salt(self) -> None:
        """Generate a new salt for key derivation."""
        salt = os.urandom(16)
        self.config.salt_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.salt_path.write_bytes(salt)

    def _load_existing_salt(self) -> None:
        """Load an existing salt."""
        self._salt = self.config.salt_path.read_bytes()

    def derive_key(self, password: str) -> bytes:
        """Derive a key from a password using PBKDF2."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self._salt,
            iterations=self.config.iterations,
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))

    def encrypt(self, data: Union[str, bytes]) -> bytes:
        """
        Encrypt data using Fernet symmetric encryption.
        
        Args:
            data: The data to encrypt (string or bytes)
            
        Returns:
            bytes: The encrypted data
        """
        if isinstance(data, str):
            data = data.encode()
        try:
            return self._fernet.encrypt(data)
        except Exception as e:
            logging.error(f"Encryption failed: {str(e)}")
            raise

    def decrypt(self, encrypted_data: bytes) -> bytes:
        """
        Decrypt Fernet-encrypted data.
        
        Args:
            encrypted_data: The data to decrypt
            
        Returns:
            bytes: The decrypted data
        """
        try:
            return self._fernet.decrypt(encrypted_data)
        except Exception as e:
            logging.error(f"Decryption failed: {str(e)}")
            raise

    def encrypt_file(self, input_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> Path:
        """
        Encrypt a file using Fernet symmetric encryption.
        
        Args:
            input_path: Path to the file to encrypt
            output_path: Optional path for the encrypted file. If not provided, appends '.encrypted'
            
        Returns:
            Path: Path to the encrypted file
        """
        input_path = Path(input_path)
        if output_path is None:
            output_path = input_path.with_suffix(input_path.suffix + '.encrypted')
        else:
            output_path = Path(output_path)

        try:
            data = input_path.read_bytes()
            encrypted_data = self.encrypt(data)
            output_path.write_bytes(encrypted_data)
            return output_path
        except Exception as e:
            logging.error(f"File encryption failed: {str(e)}")
            raise

    def decrypt_file(self, input_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> Path:
        """
        Decrypt a Fernet-encrypted file.
        
        Args:
            input_path: Path to the encrypted file
            output_path: Optional path for the decrypted file
            
        Returns:
            Path: Path to the decrypted file
        """
        input_path = Path(input_path)
        if output_path is None:
            output_path = input_path.with_suffix('')
        else:
            output_path = Path(output_path)

        try:
            encrypted_data = input_path.read_bytes()
            decrypted_data = self.decrypt(encrypted_data)
            output_path.write_bytes(decrypted_data)
            return output_path
        except Exception as e:
            logging.error(f"File decryption failed: {str(e)}")
            raise