import unittest
from io import StringIO
import numpy as np
import dask.array as da
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
    def setUp(self):
        self.instant = Orion.Instants()

    def test_add_variable(self):
        self.instant.add_variable('var1', [[1, 2, 3], [4, 5, 6]])
        self.assertTrue(da.all(self.instant['var1'].data == da.array([[1, 2, 3], [4, 5, 6]])))

    def test_delete_variable(self):
        self.instant.add_variable('var1', [[1, 2, 3], [4, 5, 6]])
        self.instant.delete_variable('var1')
        with self.assertRaises(KeyError):
            _ = self.instant['var1']

    def test_rename_variable(self):
        self.instant.add_variable('var1', [[1, 2, 3], [4, 5, 6]])
        self.instant.rename_variable('var1', 'var2')
        self.assertTrue(da.all(self.instant['var2'].data == da.array([[1, 2, 3], [4, 5, 6]])))

    def test_compute_new_variable(self):
        self.instant.add_variable('var1', [[1, 2, 3], [4, 5, 6]])
        self.instant.add_variable('var2', [[1, 1, 1], [1, 1, 1]])
        self.instant.compute('var3 = var1 + var2')
        self.assertTrue(da.all(self.instant['var3'].data == da.array([[2, 3, 4], [5, 6, 7]])))

    def test_compute_update_existing(self):
        self.instant.add_variable('Time', [[1, 2, 3], [4, 5, 6]])
        self.instant.compute('Time = Time + 5')
        self.assertTrue(da.all(self.instant['Time'].data == da.array([[6, 7, 8], [9, 10, 11]])))

    def test_compute_scalar_result(self):
        self.instant.add_variable('var1', [[1, 2, 3], [4, 5, 6]])
        self.instant.compute('sum_var1 = var1.sum()')
        self.assertEqual(self.instant['sum_var1'].data.compute(), 21)
    
class TestZones(unittest.TestCase):
    def setUp(self):
        self.zone = Orion.Zones()

    def test_add_instant(self):
        self.zone.add_instant('instant1')
        self.assertTrue('instant1' in self.zone.keys())

    def test_delete_instant(self):
        self.zone.add_instant('instant1')
        self.zone.delete_instant('instant1')
        with self.assertRaises(KeyError):
            _ = self.zone['instant1']

    def test_rename_instant(self):
        self.zone.add_instant('instant1')
        self.zone.rename_instant('instant1', 'instant2')
        self.assertTrue('instant2' in self.zone.keys())

    def test_compute_across_instants(self):
        self.zone.add_instant('instant1')
        self.zone.add_instant('instant2')
        self.zone['instant1'].add_variable('var1', [[1, 2, 3], [4, 5, 6]])
        self.zone['instant2'].add_variable('var1', [[7, 8, 9], [10, 11, 12]])
        self.zone.compute('var2 = var1 * 2')
        self.assertTrue(da.all(self.zone['instant1']['var2'].data == da.array([[2, 4, 6], [8, 10, 12]])))
        self.assertTrue(da.all(self.zone['instant2']['var2'].data == da.array([[14, 16, 18], [20, 22, 24]])))

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
    
    def test_compute_Variable_in_itself(self):
        self.base["Zone1"]["Instant1"].add_variable("variable1", [1, 2, 3])
        self.base["Zone1"]["Instant1"].add_variable("variable2", [4, 5, 6])
        self.base.compute("variable1 = variable1 * variable1")
        expected = np.array([1*1, 2*2, 3*3])
        np.testing.assert_array_equal(self.base["Zone1"]["Instant1"]["variable1"].data, expected)
    
    def test_compute_instant_variable(self):
        self.base["Zone1"]["Instant1"].add_variable("variable1", [1, 2, 3])
        self.base["Zone1"]["Instant1"].add_variable("variable2", [4, 5, 6])
        self.base["Zone1"]["Instant2"].add_variable("variable2", [4, 5, 6])
        
        self.base["Zone2"]["Instant1"].add_variable("variable2", [4, 5, 6])
        self.base["Zone2"]["Instant2"].add_variable("variable2", [4, 5, 6])
        
        self.base.compute("variable3 = variable1 * variable1")
        expected = np.array([1*1, 2*2, 3*3])
        np.testing.assert_array_equal(self.base["Zone1"]["Instant1"]["variable3"].data, expected)
        self.assertIsNot("variable3", self.base["Zone1"]["Instant2"].keys())
        self.assertIsNot("variable3", self.base["Zone2"]["Instant1"].keys())
        self.assertIsNot("variable3", self.base["Zone2"]["Instant2"].keys())

    def test_compute_no_variable(self):
        self.base["Zone1"]["Instant1"].add_variable("variable1", [1, 2, 3])
        self.base.compute("variable2 = variable1 * 3")
        expected = np.array([1*3, 2*3, 3*3])
        np.testing.assert_array_equal(self.base["Zone1"]["Instant1"]["variable2"].data, expected)

    def test_show(self):
        # Redirect stdout to capture print statements
        for zone in self.base.keys():
            for instant in self.base[zone].keys():
                for var in ['variable1', 'variable2']:
                    self.base[zone][instant].add_variable(var, [1, 2, 3])
                
        self.base.show()
        
        self.base.show(stats=True)
        
    def test_compute_across_zones_and_instants(self):
        self.base['Zone1']['Instant1'].add_variable('var1', [[1, 2, 3], [4, 5, 6]])
        self.base['Zone2']['Instant1'].add_variable('var1', [[7, 8, 9], [10, 11, 12]])
        self.base.compute('var2 = var1 * 2')
        self.assertTrue(da.all(self.base['Zone1']['Instant1']['var2'].data == da.array([[2, 4, 6], [8, 10, 12]])))
        self.assertTrue(da.all(self.base['Zone2']['Instant1']['var2'].data == da.array([[14, 16, 18], [20, 22, 24]])))

    def test_compute_with_missing_variables(self):
        self.base['Zone1']['Instant1'].add_variable('var1', [[1, 2, 3], [4, 5, 6]])
        self.base.compute('var2 = var1 + missing_var')
        # var2 should not be created in any instant due to missing variable
        with self.assertRaises(KeyError):
            _ = self.base['Zone1']['Instant1']['var2']
        with self.assertRaises(KeyError):
            _ = self.base['Zone2']['Instant1']['var2']
            
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
    
    def test_init_zone_instant(self):
        zones = [f"zone_{_}" for _ in range(2)]
        instants = [f"instant_{_}" for _ in range(2)]
        self.base.init(zones, instants)
        self.assertEqual(list(self.base.keys()), zones)
        for zone in zones:
            self.assertEqual(list(self.base[zone].keys()), instants)

    def test_init_zone_instant_variables(self):
        zones = [f"zone_{_}" for _ in range(2)]
        instants = [f"instant_{_}" for _ in range(2)]
        variables_name = [f"variable_{_}" for _ in range(2)]
        variables_values = 2*[[i for i in np.arange(5)]]
        self.base.init(zones, instants, variables_name, variables_values)
        self.assertEqual(list(self.base.keys()), zones)
        for zone in zones:
            self.assertEqual(list(self.base[zone].keys()), instants)
            for instant in instants:
                self.assertEqual(list(self.base[zone][instant].keys()), variables_name)
    
    def test_add_zone_instant_variables(self):
        # Add after init base
        self.setUp()
        zones = [f"zone_{_}" for _ in range(2)]
        instants = [f"instant_{_}" for _ in range(2)]
        variables_names = [f"variable_{_}" for _ in range(2)]
        variables_values = 2*[[i for i in np.arange(5)]]
        self.base.init(zones, instants, variables_names, variables_values)
        
        instantsbis = [f"instantbis_{_}" for _ in range(2)]
        variables_namesbis = [f"variablebis_{_}" for _ in range(2)]
        variables_valuesbis = 2*[[i for i in np.arange(5)]]
        self.base.add(zones, instantsbis, variables_namesbis, variables_valuesbis)

        self.assertEqual(list(self.base.keys()), zones)
        for zone in zones:
            self.assertEqual(list(self.base[zone].keys()), instants + instantsbis)
            for instant in instants:
                self.assertEqual(list(self.base[zone][instant].keys()), variables_names)
            for instant in instantsbis:
                self.assertEqual(list(self.base[zone][instant].keys()), variables_namesbis)

        # Add after init and add different zones, instants and variables
        self.setUp()
        zones = [f"zone_{_}" for _ in range(2)]
        instants = [f"instant_{_}" for _ in range(2)]
        variables_names = [f"variable_{_}" for _ in range(2)]
        variables_values = 2*[[i for i in np.arange(5)]]
        self.base.init(zones, instants, variables_names, variables_values)
        
        zonesbis = [f"zonebis_{_}" for _ in range(2)]
        instantsbis = [f"instantbis_{_}" for _ in range(2)]
        variables_namesbis = [f"variablebis_{_}" for _ in range(2)]
        variables_valuesbis = 2*[[i for i in np.arange(5)]]
        self.base.add(zonesbis, instantsbis, variables_namesbis, variables_valuesbis)

        self.assertEqual(list(self.base.keys()), zones + zonesbis)
        for zone in zones:
            self.assertEqual(list(self.base[zone].keys()), instants)
            for instant in instants:
                self.assertEqual(list(self.base[zone][instant].keys()), variables_names)
        for zone in zonesbis:
            self.assertEqual(list(self.base[zone].keys()), instantsbis)
            for instant in instantsbis:
                self.assertEqual(list(self.base[zone][instant].keys()), variables_namesbis)
                
        
        
if __name__ == "__main__":
    unittest.main()
