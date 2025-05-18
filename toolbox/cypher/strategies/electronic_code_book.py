from Crypto. Cipher import AES
from Crypto. Util.Padding import pad, unpad
import base64

from toolbox.cypher.base.cypher_base import CypherBase

class EcbEncryptionStrategy(CypherBase):

    def __init__(self):
        super().__init__()
        
    def encrypt(self, data: str, key: bytes) -> str:
        key = self._ensure_bytes(key)
        cipher = AES.new(key, AES.MODE_ECB)
        padded_data = pad(data.encode('utf-8'), AES.block_size)
        encrypted_data = cipher.encrypt(padded_data)
        
        return base64.b64encode (encrypted_data).decode('utf-8')

    def decrypt(self, data: str, key: bytes) -> str:
        key = self._ensure_bytes(key)
        cipher = AES.new(key, AES.MODE_ECB)
        encrypted_data = base64.b64decode(data)
        try:
            decrypted_padded_data = cipher.decrypt(encrypted_data) 
            return unpad(decrypted_padded_data, AES.block_size).decode ('utf-8')
        except:
            return cipher.decrypt(encrypted_data).strip().decode('utf-8')    