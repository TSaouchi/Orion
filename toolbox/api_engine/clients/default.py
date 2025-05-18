from toolbox.api_engine.base.rest_api_base import RestBaseAPI
from toolbox.api_engine.handler.rest_api import RestApiClient

class DefaultRestApiClient(RestBaseAPI):

    def __init__ (self, base_url: str, timeout: int = 30,
                  default_headers: dict = None, ssl_context: object = None):

        self.http_client = RestApiClient(base_url, timeout, default_headers, ssl_context)

    async def fetch(self, endpoint: str, params: dict = None, headers: dict = None):
        return await self.http_client.get(endpoint, params=params, headers=headers)

    async def post(self, endpoint: str, data: dict = None, params: dict = None, headers: dict = None):
        return await self.http_client.post(endpoint, params=params, data=data, headers=headers)

    async def update(self, endpoint: str, data: dict = None, params: dict = None, method_type: str = 'PUT', headers: dict = None, ssl_context: object = None):
        if method_type == "PUT":
            return await self.http_client.put(endpoint, params=params, data=data, headers=headers, ssl_context=ssl_context)
        elif method_type == "PATCH":
            return await self.http_client.patch(endpoint, params=params, data=data, headers=headers, ssl_context=ssl_context)
        raise ValueError("Invalid method for update. Use 'PUT' or 'PATCH'.")

    async def delete(self, endpoint: str, params: dict = None, headers: dict = None, ssl_context: object = None):
        return await self.http_client.delete(endpoint, params=params, headers=headers, ssl_context=ssl_context)

    async def head(self, endpoint: str, params: dict = None, headers: dict = None, ssl_context: object = None):
        return await self.http_client.head(endpoint, params=params, headers=headers, ssl_context=ssl_context)

    async def options(self, endpoint: str, params: dict = None, headers: dict = None, ssl_context: object = None):
        return await self.http_client.options (endpoint, params=params, headers=headers, ssl_context=ssl_context)

    async def trace(self, endpoint: str, params: dict = None, headers: dict = None, ssl_context: object = None):
        return await self.http_client.trace(endpoint, params=params, headers=headers, ssl_context=ssl_context)