from pymongo.errors import PyMongoError
from toolbox.mongodb.connection import MongoConnectionPool

# ---------------------------
# Query Executor (Asynchronous)
# ---------------------------
class MongoQueryExecutor:
    """
    Executes read queries on MongoDB with efficient batch processing.
    """
    def __init__(self, connection_pool: MongoConnectionPool, db_name: str):
        self.db = connection_pool.get_database(db_name)

    async def find(self, collection_name: str, query: dict, projection: dict = None,
             limit: int = 0, skip: int = 0, sort: list = None, batch_size: int = 1000) -> list:
        """
        Executes a find query with optional projection, sorting, pagination, and batch size.
        """
        try:
            collection = self.db[collection_name]
            cursor = collection.find(query, projection=projection).batch_size(batch_size)
            if sort:
                cursor = cursor.sort(sort)
            if skip:
                cursor = cursor.skip(skip)
            if limit:
                cursor = cursor.limit(limit)
            return await cursor.to_list(length=None)  # Motor uses async cursor methods
        except PyMongoError as e:
            raise Exception(f"Error executing find query: {e}")
