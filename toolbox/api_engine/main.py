import asyncio
from toolbox.api_engine.handler.api_manager import ApiManager
from toolbox.api_engine.clients.default import DefaultRestApiClient

async def main():
    api_manger = ApiManager()

    api_manger.register_client(
        name = "<client_name>",
        client = DefaultRestApiClient(
            base_url="<base_url>",
        )
    )
    
    response = await api_manger.execute(
                    client_name = "<client_name>",
                    method = "<methd>",
                    endpoint = "<endpoint>",
                    params = {
                        "<param_name>": "<param_value>",
                    }
                )

if __name__ == "__main__":
    asyncio.run(main())