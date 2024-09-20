import unittest
import numpy as np
import dask.array as da
from unittest.mock import patch, MagicMock

import Core as Orion
from DataProcessor import Processor

class TestProcessor(unittest.TestCase):

    def setUp(self):
        # Create a base using Orion.Base
        self.base = Orion.Base()
        self.base.init(zones=['zone1', 'zone2'], instants=['t1', 't2'])
        self.base['zone1']['t1'].add_variable('var1', da.from_array(np.random.randint(0, 100, size=(30, 50))))
        self.base['zone1']['t1'].add_variable('var2', da.from_array(np.random.randint(0, 100, size=(30, 50))))
        self.time_name = Orion.DEFAULT_TIME_NAME[0]
        self.base['zone1']['t1'].add_variable(self.time_name, da.from_array(np.random.randint(0, 100, size=(30, 50))))
        self.base['zone2']['t1'].add_variable('var1', da.from_array(np.random.randint(0, 100, size=(30, 50))))
        self.processor = Processor(self.base)

    def test_fusion(self):
        # Create additional bases for fusion
        base2 = Orion.Base()
        base2.init(zones=['zone1'], instants=['t1'])
        base2['zone1']['t1'].add_variable('var3', da.from_array(np.random.randint(0, 100, size=(30, 50))))

        result = Processor([self.base, base2]).fusion()

        self.assertIn('zone1', result.keys())
        self.assertIn('t1', result['zone1'].keys())
        self.assertIn('var1', result['zone1']['t1'].keys())
        self.assertIn('var2', result['zone1']['t1'].keys())
        self.assertIn('var3', result['zone1']['t1'].keys())
        self.assertIn(self.time_name, result['zone1']['t1'].keys())
        
        self.assertNotIn('var3', result['zone2']['t1'].keys())

    def test_fft(self):
        result = Processor(self.base).fft(decomposition_type = "both")

        self.assertIn('zone1', result.keys())
        self.assertIn('t1', result['zone1'].keys())
        self.assertIn('var1_real', result['zone1']['t1'].keys())
        self.assertIn('var1_img', result['zone1']['t1'].keys())
        self.assertIn('var1_phase', result['zone1']['t1'].keys())
        self.assertIn('var1_mag', result['zone1']['t1'].keys())
        
        self.assertNotIn(self.time_name, result['zone1']['t1'].keys())
        
        time_step_name = Orion.DEFAULT_TIMESTEP_NAME[0]
        self.base['zone2']['t1'].set_attribute(time_step_name,  0.1)

        result = Processor(self.base).fft(decomposition_type = "both")

        self.assertIn('zone1', result.keys())
        self.assertIn('t1', result['zone1'].keys())
        self.assertIn('var1_real', result['zone1']['t1'].keys())
        self.assertIn('var1_img', result['zone1']['t1'].keys())
        self.assertIn('var1_phase', result['zone1']['t1'].keys())
        self.assertIn('var1_mag', result['zone1']['t1'].keys())
        
        self.assertNotIn(self.time_name, result['zone1']['t1'].keys())

    def test_psd(self):

        result = Processor(self.base).psd()

        frequency_name = Orion.DEFAULT_FREQUENCY_NAME[0]
        time_name = Orion.DEFAULT_TIME_NAME[0]
        self.assertIn('zone1', result.keys())
        self.assertIn('t1', result['zone1'].keys())
        self.assertIn('var1', result['zone1']['t1'].keys())
        self.assertIn(frequency_name, result['zone1']['t1'].keys())
        self.assertNotIn(time_name, result['zone1']['t1'].keys())
        

    def test_filter(self):
        
        result = Processor(self.base).filter(cutoff = 2)

        self.assertIn('zone1', result.keys())
        self.assertIn('t1', result['zone1'].keys())
        self.assertIn('var1', result['zone1']['t1'].keys())
        self.assertEqual(len(result['zone1']['t1']['var1'].compute()), 
                         len(self.base['zone1']['t1']['var1'].compute()))

    def test_reduce(self):
        result = Processor(self.base).reduce(factor=2)

        self.assertIn('zone1', result.keys())
        self.assertIn('t1', result['zone1'].keys())
        self.assertIn('var1', result['zone1']['t1'].keys())
        
        self.assertEqual(len(result['zone1']['t1']['var1'].compute()), 
                         len(self.base['zone1']['t1']['var1'].compute())//2)
        self.assertNotEqual(len(result['zone1']['t1']['var1'].compute()), 
                         len(self.base['zone1']['t1']['var1'].compute()))

    def test_detrend(self):
        
        result = Processor(self.base).detrend()

        self.assertIn('zone1', result.keys())
        self.assertIn('t1', result['zone1'].keys())
        self.assertIn('var1', result['zone1']['t1'].keys())
        
        self.assertEqual(len(result['zone1']['t1']['var1'].compute()), 
                         len(self.base['zone1']['t1']['var1'].compute()))

    def test_smooth(self):

        result = Processor(self.base).smooth()

        self.assertIn('zone1', result.keys())
        self.assertIn('t1', result['zone1'].keys())
        self.assertIn('var1', result['zone1']['t1'].keys())
        
        self.assertEqual(len(result['zone1']['t1']['var1'].compute()), 
                         len(self.base['zone1']['t1']['var1'].compute()))

    def test_linear_regression(self):

        result = Processor(self.base).linear_regression()

        self.assertIn('zone1', result.keys())
        self.assertIn('t1', result['zone1'].keys())
        self.assertIn('var1', result['zone1']['t1'].keys())
        self.assertIn('slope', result['zone1']['t1']['var1']._attributes)
        self.assertIn('intercept', result['zone1']['t1']['var1']._attributes)

if __name__ == '__main__':
    unittest.main()