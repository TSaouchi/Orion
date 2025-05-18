import motor.motor_asyncio
from pymongo.errors import PyMongoError
from toolbox.mongodb.authentication import MongoAuth

# ---------------------------
# Connection Pool Manager using Motor (Asynchronous)
# ---------------------------
class MongoConnectionPool:
    """
    Manages the MongoClient instance with tuned connection pool parameters.
    Now using Motor for async operations.
    """
    def __init__(self, auth: MongoAuth, db_params: dict = None,
                 maxPoolSize: int = 200, minPoolSize: int = 20,
                 waitQueueTimeoutMS: int = 1000, connectTimeoutMS: int = 3000,
                 socketTimeoutMS: int = 5000, **kwargs):
        self.uri = auth.get_connection_uri()
        # Pool parameters combined into one dictionary
        pool_kwargs = {
            'maxPoolSize': maxPoolSize,
            'minPoolSize': minPoolSize,
            'waitQueueTimeoutMS': waitQueueTimeoutMS,
            'connectTimeoutMS': connectTimeoutMS,
            'socketTimeoutMS': socketTimeoutMS,
        }
        pool_kwargs.update(kwargs)
        try:
            self.client = motor.motor_asyncio.AsyncIOMotorClient(self.uri, **pool_kwargs)
        except PyMongoError as e:
            raise Exception(f"Error creating MongoClient: {e}")
        self.db_params = db_params or {}

    def get_database(self, db_name: str):
        """
        Returns the requested database. If no name is provided, uses one from db_params.
        """
        if not db_name and 'default_db' in self.db_params:
            db_name = self.db_params['default_db']
        if not db_name:
            raise ValueError("Database name must be provided either as argument or in db_params")
        return self.client[db_name]