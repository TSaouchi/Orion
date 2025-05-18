import threading

# ---------------------------
# Authentication Manager (Singleton)
# ---------------------------
class MongoAuth:
    """
    Singleton class to manage MongoDB authentication.
    Thread-safe initialization.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(MongoAuth, cls).__new__(cls)
            return cls._instance

    def __init__(self, username: str = None, password: str = None,
                 host: str = 'localhost', port: int = 27017, auth_db: str = 'admin'):
        if not hasattr(self, '_initialized'):
            self.username = username
            self.password = password
            self.host = host
            self.port = port
            self.auth_db = auth_db
            self._initialized = True

    @classmethod
    def reset_instance(cls):
        """
        Reset the singleton instance to allow reconfiguration.
        """
        with cls._lock:
            cls._instance = None

    def get_connection_uri(self) -> str:
        """
        Constructs the MongoDB connection URI.
        """
        if self.username and self.password:
            return f"mongodb://{self.username}:{self.password}@{self.host}:{self.port}/?authSource={self.auth_db}"
        return f"mongodb://{self.host}:{self.port}/"