import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import time

class Plotter:
    def __init__(self, data):
        self.data = data
        self.app = dash.Dash(__name__)

    def create_layout(self):
        return html.Div([
            html.H1("Data Visualization Dashboard"),
            
            html.Div([
                dcc.Dropdown(
                    id='zone-dropdown',
                    options=[{'label': 'All', 'value': 'All'}] + [{'label': zone, 'value': zone} for zone in self.data.keys()],
                    value=['All'], multi=True, placeholder="Select Zones"
                ),
                dcc.Dropdown(
                    id='instant-dropdown',
                    multi=True, placeholder="Select Instants", value=['All']
                ),
                dcc.Dropdown(
                    id='x-variable',
                    placeholder="Select X Variable"
                ),
                dcc.Dropdown(
                    id='y-variable',
                    placeholder="Select Y Variable"
                ),
            ], style={'width': '48%', 'display': 'inline-block'}),

            dcc.Graph(id='variable-plot', style={'height': '800px', 'margin-top': '50px'}),
            html.Div(id='stats-table')
        ])

    def get_available_variables(self, selected_zones, selected_instants):
        if 'All' in selected_zones:
            selected_zones = list(self.data.keys())
        
        if 'All' in selected_instants:
            selected_instants = set()
            for zone in selected_zones:
                selected_instants.update(self.data[zone].keys())
        
        variables = set()
        for zone in selected_zones:
            for instant in selected_instants:
                if instant in self.data[zone]:
                    variables.update(self.data[zone][instant].keys())
        
        return list(variables)

    def update_instants(self, selected_zones):
        if not selected_zones:
            return []

        if 'All' in selected_zones:
            selected_zones = list(self.data.keys())

        instants = set()
        for zone in selected_zones:
            instants.update(self.data[zone].keys())

        return [{'label': 'All', 'value': 'All'}] + [{'label': instant, 'value': instant} for instant in instants]

    def update_graph_and_stats(self, selected_zones, selected_instants, x_var, y_var):
        start_time = time.time()
        fig = go.Figure()

        if not selected_zones or not x_var or not y_var:
            return fig, None

        if 'All' in selected_zones:
            selected_zones = list(self.data.keys())
        if 'All' in selected_instants:
            selected_instants = set()
            for zone in selected_zones:
                selected_instants.update(self.data[zone].keys())
        selected_instants = list(selected_instants)

        # Plotting logic
        for zone in selected_zones:
            for instant in selected_instants:
                if instant in self.data[zone]:
                    x_data = self.data[zone][instant].get(x_var, [])
                    y_data = self.data[zone][instant].get(y_var, [])
                    if len(x_data) > 0 and len(y_data) > 0:
                        fig.add_trace(go.Scatter(
                            x=x_data,
                            y=y_data,
                            mode='markers',
                            name=f"{zone} - {instant}"
                        ))

        # Update layout
        fig.update_layout(
            xaxis_title=x_var,
            yaxis_title=y_var,
            title='Variables Plot'
        )

        end_time = time.time()
        print(f"Time taken to update graph: {end_time - start_time:.2f}s")

        return fig, None

    def setup_callbacks(self):
        @self.app.callback(
            Output('instant-dropdown', 'options'),
            Input('zone-dropdown', 'value')
        )
        def update_instants_callback(selected_zones):
            return self.update_instants(selected_zones)

        @self.app.callback(
            [Output('x-variable', 'options'),
             Output('y-variable', 'options')],
            [Input('zone-dropdown', 'value'),
             Input('instant-dropdown', 'value')]
        )
        def update_variable_dropdowns_callback(selected_zones, selected_instants):
            available_variables = self.get_available_variables(selected_zones, selected_instants)
            options = [{'label': var, 'value': var} for var in available_variables]
            return options, options  # Same options for both x and y variables

        @self.app.callback(
            Output('variable-plot', 'figure'),
            Output('stats-table', 'children'),
            [Input('zone-dropdown', 'value'),
             Input('instant-dropdown', 'value'),
             Input('x-variable', 'value'),
             Input('y-variable', 'value')]
        )
        def update_graph_and_stats_callback(selected_zones, selected_instants, x_var, y_var):
            return self.update_graph_and_stats(selected_zones, selected_instants, x_var, y_var)

    def dash(self):
        self.app.layout = self.create_layout()
        self.setup_callbacks()
        self.app.run_server(debug=True)


def generate_sample_data(num_zones=5, n=1000):
    data = {}
    for zone_id in range(num_zones):
        zone_name = f"Zone {zone_id + 1}"
        data[zone_name] = {}
        for instant_id in range(10):
            instant_name = f"Instant {instant_id + 1}"
            data[zone_name][instant_name] = {
                'X': np.random.rand(int(n)),
                'Y1': np.random.rand(int(n)),
                'Y2': np.random.rand(int(n)),
                'Z1': np.random.rand(int(n)),
                'Z2': np.random.rand(int(n)),
            }
    return data

if __name__ == '__main__':
    data = generate_sample_data()
    plotter = Plotter(data)
    plotter.dash()
