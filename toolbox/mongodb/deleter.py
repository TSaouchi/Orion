from pymongo.errors import PyMongoError
from toolbox.mongodb.connection import MongoConnectionPool

# ---------------------------
# Data Deleter with Extended Deletion and Collection Drop Logic (Asynchronous)
# ---------------------------
class MongoDataDeleter:
    """
    Handles deletion operations on MongoDB, including dropping empty collections.
    """
    def __init__(self, connection_pool: MongoConnectionPool, db_name: str):
        self.db = connection_pool.get_database(db_name)

    async def delete_one(self, collection_name: str, query: dict):
        """
        Deletes a single document matching the query.
        """
        collection = self.db[collection_name]
        try:
            result = await collection.delete_one(query)
            return result.deleted_count
        except PyMongoError as e:
            raise Exception(f"Error deleting document: {e}")

    async def delete_many(self, collection_name: str, query: dict):
        """
        Deletes multiple documents matching the query.
        """
        collection = self.db[collection_name]
        try:
            result = await collection.delete_many(query)
            return result.deleted_count
        except PyMongoError as e:
            raise Exception(f"Error deleting multiple documents: {e}")

    async def drop_collection(self, collection_name: str):
        """
        Drops an entire collection.
        """
        try:
            await self.db.drop_collection(collection_name)
            print(f"Collection '{collection_name}' dropped successfully.")
        except PyMongoError as e:
            raise Exception(f"Error dropping collection: {e}")

    async def drop_empty_collections(self):
        """
        Drops all empty collections in the database.
        """
        try:
            for collection_name in await self.db.list_collection_names():
                if await self.db[collection_name].estimated_document_count() == 0:
                    await self.drop_collection(collection_name)
        except PyMongoError as e:
            raise Exception(f"Error dropping empty collections: {e}")