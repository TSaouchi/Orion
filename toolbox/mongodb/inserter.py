import motor.motor_asyncio
import datetime
from pymongo.errors import PyMongoError, BulkWriteError
from toolbox.mongodb.connection import MongoConnectionPool

# ---------------------------
# Data Inserter with Bulk Write and Index Management (Asynchronous)
# ---------------------------
class MongoDataInserter:
    """
    Handles insert operations on MongoDB. Supports single inserts,
    bulk inserts (via bulk_write), and index management.
    """
    def __init__(self, connection_pool: MongoConnectionPool, db_name: str):
        self.db = connection_pool.get_database(db_name)

    async def insert_one(self, collection_name: str, document: dict, ttl: int = None, **kwargs):
        """
        Inserts a single document.
        """
        if ttl is not None:
            document['expires_at'] = datetime.datetime.utcnow() + datetime.timedelta(seconds=ttl)
        collection = self.db[collection_name]
        try:
            result = await collection.insert_one(document, **kwargs)
            return result.inserted_id
        except PyMongoError as e:
            raise Exception(f"Error inserting document: {e}")

    async def insert_many(self, collection_name: str, documents: list, ttl: int = None, **kwargs):
        """
        Inserts multiple documents.
        """
        if ttl is not None:
            for doc in documents:
                doc['expires_at'] = datetime.datetime.utcnow() + datetime.timedelta(seconds=ttl)
        collection = self.db[collection_name]
        try:
            result = await collection.insert_many(documents, **kwargs)
            return result.inserted_ids
        except PyMongoError as e:
            raise Exception(f"Error inserting multiple documents: {e}")

    async def bulk_insert(self, collection_name: str, documents: list, ttl: int = None):
        """
        Performs a bulk insert operation using bulk_write.
        """
        if ttl is not None:
            for doc in documents:
                doc['expires_at'] = datetime.datetime.utcnow() + datetime.timedelta(seconds=ttl)
        collection = self.db[collection_name]
        operations = [motor.motor_asyncio.InsertOne(doc) for doc in documents]
        try:
            result = await collection.bulk_write(operations, ordered=False)
            return result.inserted_count
        except BulkWriteError as bwe:
            raise Exception(f"Bulk write error: {bwe.details}")

    async def create_ttl_index(self, collection_name: str, field_name: str = 'expires_at', expireAfterSeconds: int = 0):
        """
        Creates a TTL index on the specified field.
        """
        collection = self.db[collection_name]
        try:
            await collection.create_index(field_name, expireAfterSeconds=expireAfterSeconds)
            print(f"TTL index created on '{collection_name}.{field_name}' with expireAfterSeconds={expireAfterSeconds}.")
        except PyMongoError as e:
            raise Exception(f"Error creating TTL index: {e}")

    async def ensure_index(self, collection_name: str, index_fields, index_name: str = None):
        """
        Ensures an index exists on the specified field(s).
        index_fields can be a string (for a single field) or a list of tuples for compound indexes.
        """
        collection = self.db[collection_name]
        try:
            if isinstance(index_fields, list):
                await collection.create_index(index_fields, name=index_name)
            elif isinstance(index_fields, str):
                await collection.create_index([(index_fields, 1)], name=index_name)  # Ascending index
            else:
                raise ValueError("index_fields must be either a string or a list of tuples")
        except PyMongoError as e:
            raise Exception(f"Error ensuring index on {collection_name}: {e}")
    async def drop_index(self, collection_name: str, index_name: str):
        """
        Drops an index by name.
        """
        collection = self.db[collection_name]
        try:
            await collection.drop_index(index_name)
            print(f"Index '{index_name}' dropped from '{collection_name}'.")
        except PyMongoError as e:
            raise Exception(f"Error dropping index: {e}")