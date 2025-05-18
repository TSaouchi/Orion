class ApiManager:
    def __init__(self):
        self.clients: dict = {}

    def register_client(self, name: str, client):
        """Register a new API client with a unique name"""
        self.clients[name] = client

    def get_client(self, name: str):
        """Get a registered API client by name"""
        if name not in self.clients:
            raise ValueError(f"API client '{name}' is not registered.")
        return self.clients[name]

    async def execute(self, client_name: str, method: str, endpoint: str, **kwargs):
        """Execute an API request using the specified client and method"""
        client = self.get_client(client_name)

        if hasattr(client, method):
            method_func = getattr(client, method)
            return await method_func(endpoint, **kwargs)

        raise ValueError(f"Invalid method '{method}'")