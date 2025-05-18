from pathlib import Path
from domain.stocks.handler.database_handler import DatabaseHandler

from utils.file_reader_utils import FileReader

# from toolbox.sqlit.authentication import SqlitAuth as Auth
# from toolbox.sqlit.connection import SqlitConnectionPool as ConnectionPool
# from toolbox.sqlit.deleter import SqlitDataDeleter as Deleter
# from toolbox.sqlit.inserter import SqlitDataInserter as Inserter
# from toolbox.sqlit.query_executor import SqlitQueryExecutor as QueryExecutor


# class SqlitHandler(DatabaseHandler):
    
#     def authenticate(self):
#         self.auth = Auth()

#     def connect(self):
#         config_path = Path(__file__).parent / "conf" / "config.json"
#         database_conf =  FileReader(config_path).read()["datebase"]
#         self.connection_pool = ConnectionPool(self.auth, 
#                                               {"db_name" : 
#                                                   database_conf["db_name"]})

#     def delete(self, query):
#         self.deleter = Deleter(self.connection_pool)

#     def query(self, query):
#         self.executor = QueryExecutor(self.connection_pool)

#     def insert(self, data):
#         self.inserter = Inserter(self.connection_pool)