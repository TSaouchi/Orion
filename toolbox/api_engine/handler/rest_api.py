import aiohttp
import logging
logger = logging.getLogger()

class RestApiClient:

    def __init__(self, base_url: str, timeout: int = 30, default_headers: dict = None, ssl_context: object = None):
        self.base_url = base_url
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.ssl_context = ssl_context
        self.default_headers = default_headers or {
            'Content-Type': 'application/json',
            'Accept': 'application/json'}


    async def _make_request(self, method: str, endpoint: str, params: dict = None, 
                            data: dict = None, headers: dict = None, 
                            timeout: int = None, ssl_context: object = None): 
        
        request_headers = {**self.default_headers, **(headers or {})}
        timeout = aiohttp.ClientTimeout(total=timeout) if timeout else self.timeout

        async with aiohttp.ClientSession(timeout=timeout) as session:
            url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
            logger.debug(f"Making {method} request to {url}")
            
            kwargs = {
                "method" : method, 
                "url" : url, 
                "params" : params, 
                "data" : data, 
                "headers" : request_headers, 
                "ssl" : self.ssl_context
            }    
            
            async with session.request(**kwargs) as response:
                logger.info(f"Method: {kwargs["method"]} Response status {response.status}\n\tURL {response.url}")
                response.raise_for_status()
                if response.content_type == "application/json":
                    return await response.json()
                return await response.text()

    async def get(self, endpoint: str, params: dict = None, headers: dict = None): 
        return await self._make_request('GET', endpoint, params=params, headers=headers)

    async def post(self, endpoint: str, data: dict = None, params: dict = None, headers: dict = None): 
        return await self._make_request('POST', endpoint, params=params, data=data, headers=headers)

    async def put(self, endpoint: str, data: dict = None, params: dict = None, headers: dict = None):
        return await self._make_request('PUT', endpoint, params=params, data=data, headers=headers)

    async def patch(self, endpoint: str, data: dict = None, params: dict = None, headers: dict = None): 
        return await self._make_request('PATCH', endpoint, params=params, data=data, headers=headers)

    async def delete(self, endpoint: str, params: dict = None, headers: dict = None, ssl_context: object = None):
        return await self._make_request('DELETE', endpoint, params=params, headers=headers, ssl_context=ssl_context)

    async def head(self, endpoint: str, params: dict = None, headers: dict = None, ssl_context: object = None):
        return await self._make_request('HEAD', endpoint, params=params, headers=headers, ssl_context=ssl_context)

    async def options(self, endpoint: str, params: dict = None, headers: dict = None, ssl_context: object = None):
        return await self._make_request('OPTIONS', endpoint, params=params, headers=headers, ssl_context=ssl_context)

    async def trace(self, endpoint: str, params: dict = None, headers: dict = None, ssl_context: object = None):
        return await self._make_request('TRACE', endpoint, params=params, headers=headers, ssl_context=ssl_context)

