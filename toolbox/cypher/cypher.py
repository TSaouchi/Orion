from enum import Enum
import os
from toolbox.cypher.base.cypher_base import CypherBase
from toolbox.cypher.strategies.electronic_code_book import EcbEncryptionStrategy
from toolbox.cypher.strategies.cipher_block_chaining import CbcEncryptionStrategy

class EncryptionStrategyType(Enum):
    ECB = "ECB"
    CBC = "CBC"

class Cypher(CypherBase):

    def __init__(self, encryption_key, strategy: CypherBase = None):
        self.encryption_key = encryption_key
        
        match strategy:
            case EncryptionStrategyType.ECB.value:
                self.strategy = EcbEncryptionStrategy()
            case _:
                self.strategy = CbcEncryptionStrategy()
                
    def encrypt(self, my_str: str) -> str:
        return self.strategy.encrypt(my_str, self.encryption_key)

    def decrypt(self, my_str: str) -> str:
        return self.strategy.decrypt(my_str, self.encryption_key)

if __name__ == "__main__":
    my_secret_key = os.environ.get("cypher_key")
    
    cypher = Cypher(my_secret_key)
    encoded_str = cypher.encrypt("Hello, World!")
    print(encoded_str)
    decode = cypher.decrypt(encoded_str)
    print(decode)