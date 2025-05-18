from abc import ABC, abstractmethod

class DatabaseHandler(ABC):
    
    @abstractmethod
    def authenticate(self):
        raise NotImplementedError
    
    @abstractmethod
    def connect(self):
        raise NotImplementedError
    
    @abstractmethod
    def delete(self, query):
        raise NotImplementedError
    
    @abstractmethod
    def query(self, query):
        raise NotImplementedError
    
    @abstractmethod
    def insert(self, data):
        raise NotImplementedError