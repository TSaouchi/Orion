from Crypto.Protocol.KDF import PBKDF2
from Crypto.Hash import SHA256
from socket import gethostname
from abc import ABC, abstractmethod

class CypherBase(ABC):
    
    @abstractmethod
    def encrypt(self, data: str, key: bytes) -> str:
        raise NotImplementedError
    
    @abstractmethod
    def decrypt(self, data: str, key: bytes) -> str:
        raise NotImplementedError
    
    @staticmethod
    def _ensure_bytes(key):
        if isinstance(key, str):
            key_material = key.encode('utf-8')
        elif isinstance(key, bytes):
            key_material = key
        else:
            raise TypeError("Encryption key must be a string or bytes")
        
        salt = gethostname()
        derived_key = PBKDF2(key_material, salt, dkLen=32, count=1000, hmac_hash_module=SHA256)
        return derived_key