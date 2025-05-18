import os
import asyncio
import aiohttp

import core as Orion

# Singleton MetaClass for ensuring a single instance
class SingletonMeta(type):
    _instances = {}
    _lock = asyncio.Lock()

    async def __call__(cls, *args, **kwargs):
        async with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]

# Financial Data Fetcher with Singleton pattern
class FinancialDataFetcher(metaclass=SingletonMeta):
    def __init__(self, api_key_filename=None):
        # self.api_key = self.__load_api_key(api_key_filename)
        self.api_base = self.__construct_api_base()
        self.cache = {}

    async def fetch_data_for_company(self, session, symbol, interval, range):
        if symbol in self.cache:
            print(f"Data for {symbol} found in cache.")
            return self.cache[symbol]
        
        # Create the url
        url = self.__construct_api_url(symbol, interval, range)
        
        try:
            async with session.get(url) as response:
                response.raise_for_status()
                data = await response.json()

                if data:
                    self.cache[symbol] = data  # Cache the result
                    return data
                else:
                    raise ValueError(f"No data found for symbol {symbol}")

        except Exception as e:
            print(f"Failed to fetch data for {symbol}: {e}")
            return None

    async def fetch_data_for_all_companies(self, symbols, intervals, ranges):
        async with aiohttp.ClientSession() as session:
            tasks = [
                self.fetch_data_for_company(session, symbol, interval, range)
                for symbol, interval, range in zip(symbols, intervals, ranges)
            ]
            responses = await asyncio.gather(*tasks)
            
        return responses
    
    # Construct the API request URL
    def __construct_api_url(self, symbol, interval, range):
        url = (
            f"{self.endpoint.value}/v8/finance/chart/{symbol}?interval={interval}&range={range}"
        )
        return url
    
    def __construct_api_base(self):
        base = Orion.Base()
        names = ['Yahoo']
        end_point = ['https://query1.finance.yahoo.com']
        parameters_names = ['symbols', 'intervals', 'periods']
        parameters_values = [['APPL', 'GOOGL'],
                             ['1m', '1m'],
                             ['1wk', '1wk']]
        base.init(names, end_point, parameters_names, parameters_values)
        


#  ----------------------------------------------------------------------------------------------
    # async def cache_fetched_data(self):
    async def create_base(self, symbols, intervals, ranges):
        symbols_data = await self.fetch_data_for_all_companies(symbols, 
                                                               intervals, ranges)
        base = Orion.Base()
        
        elements = list(symbols_data[0].keys())
        for symbol_data in symbols_data: 
            zone = symbol_data[elements[0]]['2. Symbol']

            data = symbol_data[elements[1]]
            # Initialize variable names
            variable_names = ['time'] + list(data[next(iter(data))].keys())
            variable_values = [[] for _ in variable_names]

            # Populate variable values
            for date, metrics in data.items():
                # Append the date to the first variable (time)
                variable_values[0].append(date)
                
                # Append values for each metric
                for idx, key in enumerate(metrics.keys(), start=1):
                    variable_values[idx].append(float(metrics[key]))

            base.add([zone], ['instant'], variable_names, variable_values)

        return base

    
    # Load API key from file
    # def __load_api_key(self, api_key_filename):
    #     if not api_key_filename:
    #         api_key_filename = ".api_key"
    #     with open(api_key_filename, "r") as file:
    #         return file.read().strip()

# Start the Flask app
if __name__ == "__main__":
    symbols = ["AAPL", "GOOGL"]
    intervals = len(symbols)*["1m"]
    ranges = len(symbols)*["1wk"]
    async def main(symbols, intervals, ranges):
        fetcher = FinancialDataFetcher()
        return await fetcher.create_base(symbols, intervals, ranges)

    base = asyncio.run(main(symbols, intervals, ranges))
    # print(base)
    # fetcher = await FinancialDataFetcher()
    # base = await fetcher.create_base(symbols, intervals, ranges)

    plotter = Orion.Plotter(base)
    plotter.run()