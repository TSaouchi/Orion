import unittest
import numpy as np
import dask.array as da
from dash import html
import plotly.graph_objs as go
import pandas as pd
from unittest.mock import patch
import Core as Orion

# Import the classes we want to test
from PlotterDash import Plotter

class TestPlotter(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.base = self._generate_test_data()
        self.plotter = Plotter(self.base)

    def _generate_test_data(self):
        """Helper method to generate test data"""
        base = Orion.Base()
        nzone = 5
        ninstant = 5
        n = 10
        zones = [f"Zone_{i}" for i in range(nzone)]
        instants = [f"Instant_{i}" for i in range(ninstant)]
        base.init(zones, instants)
        
        var1 = [f"var_{i}" for i in range(0, 2)]
        
        for zone in zones[0:2]:
            for instant in instants:
                for var in var1:
                    base[zone][instant].add_variable(var, 
                                                     np.sin(da.random.random(n, 1))
                                                     )
        return base

    def test_init(self):
        """Test the initialization of Plotter"""
        self.assertIsNotNone(self.plotter.app)
        self.assertIsNotNone(self.plotter.cache)
        self.assertEqual(self.plotter.base, self.base)

    def test_create_layout(self):
        """Test the creation of the dashboard layout"""
        layout = self.plotter.create_layout()
        self.assertIsInstance(layout, html.Div)
        # Check if main components are present
        self.assertEqual(len(layout.children), 5)  # Title, dropdowns, graph, stats table, loading

    def test_create_dropdowns(self):
        """Test the creation of dropdown menus"""
        dropdowns = self.plotter.create_dropdowns()
        self.assertIsInstance(dropdowns, html.Div)
        # Check if all dropdown components are present
        self.assertEqual(len(dropdowns.children), 2)  # Two main div containers

    def test_get_available_variables(self):
        """Test getting available variables based on selection"""
        variables = self.plotter.get_available_variables(
            selected_zones=["Zone_0"],
            selected_instants=["Instant_0"]
        )
        self.assertEqual(set(variables), {"var_0", "var_1"})

    def test_get_selected_items_all(self):
        """Test _get_selected_items method with 'All' selection"""
        all_items = ["item1", "item2", "item3"]
        selected = self.plotter._get_selected_items(["All"], all_items)
        self.assertEqual(selected, all_items)

    def test_get_selected_items_specific(self):
        """Test _get_selected_items method with specific selection"""
        all_items = ["item1", "item2", "item3"]
        selected = self.plotter._get_selected_items(["item1", "item2"], all_items)
        self.assertEqual(selected, ["item1", "item2"])

    def test_update_instants(self):
        """Test updating instant dropdown options"""
        options = self.plotter.update_instants(["Zone_0"])
        self.assertTrue(any(opt["value"] == "All" for opt in options))
        self.assertTrue(any(opt["value"] == "Instant_0" for opt in options))
        self.assertTrue(any(opt["value"] == "Instant_1" for opt in options))

    def test_update_variable_dropdowns(self):
        """Test updating variable dropdown options"""
        x_options, y_options, z_options = self.plotter.update_variable_dropdowns(
            ["Zone_0"], ["Instant_0"]
        )
        # Check if options are properly generated
        self.assertTrue(len(x_options) > 0)
        self.assertTrue(len(y_options) > 0)
        self.assertTrue(len(z_options) > 0)
        # Check if 'All' option is present in y and z dropdowns
        self.assertTrue(any(opt["value"] == "All" for opt in y_options))
        self.assertTrue(any(opt["value"] == "All" for opt in z_options))

    def test_aggregate_data(self):
        """Test data aggregation"""
        aggregated = self.plotter.aggregate_data(
            selected_zones=["Zone_0"],
            selected_instants=["Instant_0"],
            x_var="var_0",
            y_vars=["var_1"],
            z_vars=None
        )
        self.assertIsInstance(aggregated, set)
        self.assertTrue(len(aggregated) > 0)

    def test_create_figure_2d(self):
        """Test creation of 2D figure"""
        aggregated_set = {("var_0", "Zone_0", "Instant_0"), ("var_1", "Zone_0", "Instant_0")}
        fig = self.plotter.create_figure(
            aggregated_set,
            x_var="var_0",
            y_vars=["var_1"],
            z_vars=[],
            is_3d=False
        )
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 1)  # One trace for 2D plot

    def test_create_figure_3d(self):
        """Test creation of 3D figure"""
        aggregated_set = {
            ("var_0", "Zone_0", "Instant_0"),
            ("var_1", "Zone_0", "Instant_0"),
            ("var_2", "Zone_0", "Instant_0")
        }
        fig = self.plotter.create_figure(
            aggregated_set,
            x_var="var_0",
            y_vars=["var_1"],
            z_vars=["var_2"],
            is_3d=True
        )
        self.assertIsInstance(fig, go.Figure)

    def test_create_stats_table(self):
        """Test creation of stats table"""
        test_df = pd.DataFrame({
            'Stat': ['Mean', 'Std'],
            'Value': [1.0, 0.5]
        })
        table = self.plotter.create_stats_table(test_df)
        self.assertIsNotNone(table)

    @patch('dash.Dash.run_server')
    def test_run(self, mock_run_server):
        """Test running the server"""
        self.plotter.run()
        mock_run_server.assert_called_once_with(debug=True)

if __name__ == '__main__':
    unittest.main()