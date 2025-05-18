from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
import base64

from toolbox.cypher.base.cypher_base import CypherBase

class CbcEncryptionStrategy(CypherBase):
    
    def __init__(self):
        super().__init__()

    def encrypt(self, data: str, key: bytes) -> str:
        key = self._ensure_bytes(key)
        iv = get_random_bytes (AES.block_size)
        cipher = AES.new(key, AES.MODE_CBC, iv)
        padded_data = pad(data.encode('utf-8'), AES.block_size)
        encrypted_data = cipher.encrypt(padded_data)

        return base64.b64encode(iv + encrypted_data).decode('utf-8')

    def decrypt(self, data: str, key: bytes) -> str:
        key = self._ensure_bytes(key)
        decoded_data = base64.b64decode(data)
        iv = decoded_data[:AES.block_size]
        encrypted_data = decoded_data[AES.block_size:]
        cipher = AES.new(key, AES.MODE_CBC, iv)
        decrypted_padded_data = cipher.decrypt(encrypted_data)
        
        return unpad(decrypted_padded_data, AES.block_size).decode ('utf-8')
    
    