from abc  import ABC, abstractmethod

class RestBaseAPI(ABC):

    @abstractmethod
    async def fetch(self, endpoint: str, params: dict = None, headers: dict = None):
        """Fetch data from API using GET method"""
        raise NotImplementedError

    @abstractmethod
    async def post(self, endpoint: str, data: dict = None, params: dict = None, headers: dict = None):
        """Create new resource using POST method"""
        raise NotImplementedError

    @abstractmethod
    async def update(self, endpoint: str, data: dict = None, params: dict = None, method: str = 'PUT', headers: dict = None):
        """Update resource using PUT or PATCH method"""
        raise NotImplementedError

    @abstractmethod
    async def delete(self, endpoint: str, params: dict = None, headers: dict = None):
        """Delete resource using DELETE method"""
        raise NotImplementedError

    @abstractmethod
    async def head(self, endpoint: str, params: dict = None, headers: dict = None):
        """Get resource headers using HEAD method"""
        raise NotImplementedError

    @abstractmethod
    async def options(self, endpoint: str, params: dict = None, headers: dict = None):
        """Get resource options using OPTIONS method"""
        raise NotImplementedError

    @abstractmethod
    async def trace(self, endpoint: str, params: dict = None, headers: dict = None):
        """Trace request using TRACE method"""
        raise NotImplementedError