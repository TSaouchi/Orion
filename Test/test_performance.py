import unittest
import time
import dask.array as da
import Core as Orion
from Debug import PerformanceStats

# Assuming Orion.Base() and PerformanceStats are defined in your environment
# from orion_module import Orion, PerformanceStats  # Modify this as per your setup

class TestPerformance(unittest.TestCase):
    def setUp(self):
        # Initialize the setup for zones, instants, and variables
        self.nzone = 500
        self.ninstant = 500
        self.nvar = 100
        self.n = int(1e8)
        self.zones = [f"Zone_{i}" for i in range(self.nzone)]
        self.instants = [f"Instant_{i}" for i in range(self.ninstant)]
        self.var1 = [f"var_{i}" for i in range(self.nvar)]
        self.var1_value = self.nvar * [da.random.random((self.n, 1), chunks='auto')]
        self.base = Orion.Base()  # Assuming this is the correct initialization of your base

    def test_performance_init(self):
        # Start the timer
        # Performance stats for base.init
        with PerformanceStats() as stats:
            self.base.init(self.zones, self.instants, self.var1, self.var1_value)
        
        # Performance stats for base.compute
        with PerformanceStats() as stats:
            self.base.compute('var2 = 2')
        # Assert that the total operation (init + compute) takes less than 40 seconds
        self.assertLess(stats['execution_time'], 40, f"Code took too long: {stats['execution_time']:.2f} seconds")
    
    def test_performance_compute_multithread(self):
        self.base.init(self.zones, self.instants, self.var1, self.var1_value)
        
        # Performance stats for base.compute
        with PerformanceStats() as stats:
            self.base.compute('var2 = 2', chunk_size=1000000)
        # Assert that the total operation (init + compute) takes less than 40 seconds
        self.assertLess(stats['execution_time'], 40, f"Code took too long: {stats['execution_time']:.2f} seconds")
    
    def test_performance_compute_multiprocess(self):
        self.base.init(self.zones, self.instants, self.var1, self.var1_value)
        
        # Performance stats for base.compute
        with PerformanceStats() as stats:
            self.base.compute('var2 = 2')
        # Assert that the total operation (init + compute) takes less than 40 seconds
        self.assertLess(stats['execution_time'], 10, f"Code took too long: {stats['execution_time']:.2f} seconds")

if __name__ == "__main__":
    unittest.main()
