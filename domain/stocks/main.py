import asyncio
import os
from pathlib import Path
from utils.file_reader_utils import FileReader
from toolbox.api_engine.handler.api_manager import ApiManager
from toolbox.api_engine.clients.default import DefaultRestApiClient
from toolbox.cypher.cypher import Cypher

def payloads(symbol, apis_config: dict, cypher: Cypher):
    return [
        {"client_name" : "alpha_vantage",
        "method" : "fetch",
        "endpoint" : "query",
        "params" : {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "interval": "1d",
            "apikey": cypher.decrypt(apis_config["alpha_vantage"]["api_key"])
            }
        },
        {"client_name" : "twelve_data",
        "method" : "fetch",
        "endpoint" : "time_series",
        "params" : {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "interval": "1day",
            "apikey": cypher.decrypt(apis_config["twelve_data"]["api_key"])
            }
        }
    ]

def fetch(api_manger: ApiManager, payload):
    return api_manger.execute(**payload)

async def main():
    api_manger = ApiManager()
    config_path = Path(__file__).parent / "conf" / "config.json"
    apis_config =  FileReader(config_path).read()["apis"]
    cypher_key = os.environ.get("cypher_key")
    cypher = Cypher(cypher_key)

    for key in apis_config:
        api_manger.register_client(
            name = key,
            client = DefaultRestApiClient(base_url=apis_config[key]["base_url"])
        )
    
    symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
    all_payloads = [payloads(symbol, apis_config, cypher)[1] for symbol in symbols]
    tasks = [fetch(api_manger, paylod) for paylod in all_payloads]
    response = await asyncio.gather(*tasks)
    print(response)
        
if __name__ == "__main__":
    asyncio.run(main())