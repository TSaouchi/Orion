import unittest
from io import StringIO
import numpy as np
import sys
import Core as Orion

class TestCustomAttributes(unittest.TestCase):
    def test_set_get_attribute(self):
        obj = Orion.CustomAttributes()
        obj.set_attribute('attr1', 'value1')
        self.assertEqual(obj.get_attribute('attr1'), 'value1')

    def test_delete_attribute(self):
        obj = Orion.CustomAttributes()
        obj.set_attribute('attr1', 'value1')
        obj.delete_attribute('attr1')
        with self.assertRaises(AttributeError):
            _ = obj.attr1

    def test_rename_attribute(self):
        obj = Orion.CustomAttributes()
        obj.set_attribute('attr1', 'value1')
        obj.rename_attribute('attr1', 'attr2')
        self.assertEqual(obj.get_attribute('attr2'), 'value1')

class TestVariables(unittest.TestCase):
    def test_add_variable(self):
        var = Orion.Variables([[1, 2, 3], [4, 5, 6]])
        self.assertTrue(np.array_equal(var.data, np.array([[1, 2, 3], [4, 5, 6]])))

class TestInstants(unittest.TestCase):
    def test_add_variable(self):
        instant = Orion.Instants()
        instant.add_variable('var1', [[1, 2, 3], [4, 5, 6]])
        self.assertTrue(np.array_equal(instant['var1'].data, np.array([[1, 2, 3], [4, 5, 6]])))

    def test_delete_variable(self):
        instant = Orion.Instants()
        instant.add_variable('var1', [[1, 2, 3], [4, 5, 6]])
        instant.delete_variable('var1')
        with self.assertRaises(KeyError):
            _ = instant['var1']

    def test_rename_variable(self):
        instant = Orion.Instants()
        instant.add_variable('var1', [[1, 2, 3], [4, 5, 6]])
        instant.rename_variable('var1', 'var2')
        self.assertTrue(np.array_equal(instant['var2'].data, np.array([[1, 2, 3], [4, 5, 6]])))
    
class TestZones(unittest.TestCase):
    def test_add_instant(self):
        zone = Orion.Zones()
        zone.add_instant('instant1')
        self.assertTrue('instant1' in zone.keys())

    def test_delete_instant(self):
        zone = Orion.Zones()
        zone.add_instant('instant1')
        zone.delete_instant('instant1')
        with self.assertRaises(KeyError):
            _ = zone['instant1']

    def test_rename_instant(self):
        zone = Orion.Zones()
        zone.add_instant('instant1')
        zone.rename_instant('instant1', 'instant2')
        self.assertTrue('instant2' in zone.keys())

class TestBase(unittest.TestCase):
    def setUp(self):
        self.base = Orion.Base()
        self.base.init(zones=["Zone1", "Zone2"], instants=["Instant1", "Instant2"])

    def test_add_zone(self):
        self.base.add_zone("Zone3")
        self.assertTrue("Zone3" in self.base.keys())

    def test_delete_zone(self):
        self.base.delete_zone("Zone1")
        with self.assertRaises(KeyError):
            _ = self.base["Zone1"]

    def test_rename_zone(self):
        self.base.rename_zone("Zone1", "ZoneNew")
        self.assertTrue("ZoneNew" in self.base.keys())
    
    def test_rename_variable_in_base(self):
        base = Orion.Base()
        # Add zones and instants to the base
        base.init(zones=['zone1', 'zone2'], instants=['t1', 't2'])
        # Add some variables to the instants in both zones
        base['zone1']['t1'].add_variable('var1', [[1, 2, 3], [4, 5, 6]])
        base['zone1']['t2'].add_variable('var2', [[7, 8, 9], [10, 11, 12]])
        base['zone2']['t1'].add_variable('var1', [[13, 14, 15], [16, 17, 18]])
        # Rename variables across the entire base
        base.rename_variable(['var1', 'var2'], ['new_var1', 'new_var2'])
        # Verify that the variables were renamed in all relevant zones and instants
        self.assertTrue(np.array_equal(base['zone1']['t1']['new_var1'].data, np.array([[1, 2, 3], [4, 5, 6]])))
        self.assertTrue(np.array_equal(base['zone1']['t2']['new_var2'].data, np.array([[7, 8, 9], [10, 11, 12]])))
        self.assertTrue(np.array_equal(base['zone2']['t1']['new_var1'].data, np.array([[13, 14, 15], [16, 17, 18]])))

    def test_compute(self):
        self.base["Zone1"]["Instant1"].add_variable("variable1", [1, 2, 3])
        self.base["Zone1"]["Instant1"].add_variable("variable2", [4, 5, 6])
        self.base.compute("variable3 = variable1 * variable2")
        expected = np.array([1*4, 2*5, 3*6])
        np.testing.assert_array_equal(self.base["Zone1"]["Instant1"]["variable3"].data, expected)

    def test_compute_no_variable(self):
        self.base["Zone1"]["Instant1"].add_variable("variable1", [1, 2, 3])
        self.base.compute("variable2 = variable1 * 3")
        expected = np.array([1*3, 2*3, 3*3])
        np.testing.assert_array_equal(self.base["Zone1"]["Instant1"]["variable2"].data, expected)

    def test_show(self):
        # Redirect stdout to capture print statements
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        for zone in self.base.keys():
            for instant in self.base[zone].keys():
                for var in ['variable1', 'variable2']:
                    self.base[zone][instant].add_variable(var, [1, 2, 3])
                
        self.base.show()
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout

        expected_output = (
         'Base\n  Zone: Zone1\n    Instant: Instant1\n      Variable: variable1 -> Shape :(3,)\n      Variable: variable2 -> Shape :(3,)\n    Instant: Instant2\n      Variable: variable1 -> Shape :(3,)\n      Variable: variable2 -> Shape :(3,)\n  Zone: Zone2\n    Instant: Instant1\n      Variable: variable1 -> Shape :(3,)\n      Variable: variable2 -> Shape :(3,)\n    Instant: Instant2\n      Variable: variable1 -> Shape :(3,)\n      Variable: variable2 -> Shape :(3,)\n'
        )
        self.assertEqual(output, expected_output)
        
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        self.base.show(stat=True)
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        expected_output = (
         'Base\n  Zone: Zone1\n    Instant: Instant1\n      Variable: variable1 -> Shape :(3,), stats(min, mean, max): (1, 2.0, 3)\n      Variable: variable2 -> Shape :(3,), stats(min, mean, max): (1, 2.0, 3)\n    Instant: Instant2\n      Variable: variable1 -> Shape :(3,), stats(min, mean, max): (1, 2.0, 3)\n      Variable: variable2 -> Shape :(3,), stats(min, mean, max): (1, 2.0, 3)\n  Zone: Zone2\n    Instant: Instant1\n      Variable: variable1 -> Shape :(3,), stats(min, mean, max): (1, 2.0, 3)\n      Variable: variable2 -> Shape :(3,), stats(min, mean, max): (1, 2.0, 3)\n    Instant: Instant2\n      Variable: variable1 -> Shape :(3,), stats(min, mean, max): (1, 2.0, 3)\n      Variable: variable2 -> Shape :(3,), stats(min, mean, max): (1, 2.0, 3)\n'
        )
        self.assertEqual(output, expected_output)
    
class TestBaseInit(unittest.TestCase):
    def setUp(self):
        self.base = Orion.Base()
        self.base.init()
    
    def test_init(self):
        self.assertEqual(list(self.base.keys()), Orion.DEFAULT_ZONE_NAME) 
        self.assertEqual(list(self.base[0].keys()), Orion.DEFAULT_INSTANT_NAME)
    
    def test_add(self):
        self.base.add()
        self.assertEqual(len(self.base.keys()), 1)
        self.assertEqual(len(self.base[0].keys()), 1)
        self.assertEqual(list(self.base.keys()), Orion.DEFAULT_ZONE_NAME) 
        self.assertEqual(list(self.base[0].keys()), Orion.DEFAULT_INSTANT_NAME)
        
        self.base.add(["Zone2"], ["Instant2"])
        self.assertEqual(len(self.base.keys()), 2)
        self.assertEqual(len(self.base[0].keys()), 1)
        self.assertEqual(len(self.base[1].keys()), 1)
        self.assertEqual(list(self.base.keys())[1], "Zone2") 
        self.assertEqual(list(self.base[1].keys())[0], "Instant2")
        

if __name__ == "__main__":
    unittest.main()
