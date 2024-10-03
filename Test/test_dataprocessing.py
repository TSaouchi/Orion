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
        base2['zone1']['t1'].add_variable('var3', da.from_array(np.random.randint(0, 100, size=(3, 5))))

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
    
    def test_clamp(self):
        # Set up test data
        time_values = np.array([0, 2, 4, 6, 8, 10])
        test_values = np.array([1, 5, 10, 15, 20, 25])
        
        self.base['zone1']['t1'].add_variable(self.time_name, da.from_array(time_values))
        self.base['zone1']['t1'].add_variable('test_var', da.from_array(test_values))

        # Test case 1: Clamp with both lower and upper bounds
        result = Processor(self.base).clamp(target_variable=self.time_name, clamp_band=(2, 8))
        np.testing.assert_array_equal(result['zone1']['t1'][self.time_name].compute(), np.array([2, 4, 6, 8]))
        np.testing.assert_array_equal(result['zone1']['t1']['test_var'].compute(), np.array([5, 10, 15, 20]))

        # Test case 2: Clamp with only lower bound
        result = Processor(self.base).clamp(target_variable=self.time_name, clamp_band=(6, None))
        np.testing.assert_array_equal(result['zone1']['t1'][self.time_name].compute(), np.array([6, 8, 10]))
        np.testing.assert_array_equal(result['zone1']['t1']['test_var'].compute(), np.array([15, 20, 25]))

        # Test case 3: Clamp with only upper bound
        result = Processor(self.base).clamp(target_variable=self.time_name, clamp_band=(None, 4))
        np.testing.assert_array_equal(result['zone1']['t1'][self.time_name].compute(), np.array([0, 2, 4]))
        np.testing.assert_array_equal(result['zone1']['t1']['test_var'].compute(), np.array([1, 5, 10]))

        # Test case 4: No clamping
        result = Processor(self.base).clamp(target_variable=self.time_name, clamp_band=(None, None))
        np.testing.assert_array_equal(result['zone1']['t1'][self.time_name].compute(), time_values)
        np.testing.assert_array_equal(result['zone1']['t1']['test_var'].compute(), test_values)

        # Test case 5: Empty result due to clamping
        result = Processor(self.base).clamp(target_variable=self.time_name, clamp_band=(12, 20))
        self.assertEqual(len(result['zone1']['t1'][self.time_name].data), 0)
        self.assertEqual(len(result['zone1']['t1']['test_var'].data), 0)

        # Test case 6: Clamp with reversed bounds (should handle gracefully)
        result = Processor(self.base).clamp(target_variable=self.time_name, clamp_band=(8, 2))
        np.testing.assert_array_equal(result['zone1']['t1'][self.time_name].compute(), np.array([2, 4, 6, 8]))
        np.testing.assert_array_equal(result['zone1']['t1']['test_var'].compute(), np.array([5, 10, 15, 20]))

        # Test case 7: Clamp with exact value
        result = Processor(self.base).clamp(target_variable=self.time_name, clamp_band=(4, 4))
        np.testing.assert_array_equal(result['zone1']['t1'][self.time_name].compute(), np.array([4]))
        np.testing.assert_array_equal(result['zone1']['t1']['test_var'].compute(), np.array([10]))
        
    def test_peak(self):
        # Create a simple sine wave with known peaks
        x = np.linspace(0, 4*np.pi, 200)
        y = np.sin(x)
        self.base['zone1']['t1'].add_variable(self.time_name, da.from_array(x))
        self.base['zone1']['t1'].add_variable('sine_wave', da.from_array(y))
        # Test case 1: Basic peak detection
        result = Processor(self.base).peak(height=0.5, distance=20)
        
        self.assertIn('zone1', result.keys())
        self.assertIn('t1', result['zone1'].keys())
        self.assertIn('sine_wave', result['zone1']['t1'].keys())
        
        peaks = result['zone1']['t1']['sine_wave'].compute()
        peak_times = result['zone1']['t1']['sine_wave'].get_attribute(self.time_name)
        
        # We expect 3 peaks in our sine wave
        self.assertEqual(len(peaks), 2)
        np.testing.assert_almost_equal(peaks, np.array([1, 1]), decimal=2)
        np.testing.assert_almost_equal(peak_times, np.array([np.pi/2, 5*np.pi/2]), 
                                       decimal=1)

        # Test case 2: With properties
        result = Processor(self.base).peak(height=0.5, distance=20, properties=True)
        
        peak_properties = result['zone1']['t1']['sine_wave']._attributes
        self.assertIn('peak_heights', peak_properties)

        # Test case 3: With custom dependent variable
        custom_var = np.arange(200)
        self.base['zone1']['t1'].add_variable('custom_var', da.from_array(custom_var))
        
        result = Processor(self.base).peak(dependent_variables=['custom_var'], height=0.5, distance=20)
        
        peak_custom_var = result['zone1']['t1']['sine_wave'].get_attribute('custom_var')
        self.assertEqual(len(peak_custom_var), 2)

if __name__ == '__main__':
    unittest.main()