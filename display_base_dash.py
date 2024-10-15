import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
from flask_caching import Cache
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import time
from collections import defaultdict

class Plotter:
    def __init__(self, data):
        self.data = data
        self.app = dash.Dash(__name__)
        self.cache = Cache(self.app.server, config={
            'CACHE_TYPE': 'filesystem',
            'CACHE_DIR': 'cache-directory',
            'CACHE_DEFAULT_TIMEOUT': 120  # Increased cache timeout for performance
        })

    def create_layout(self):
        return html.Div([
            html.H1("Data Visualization Dashboard"),
            self.create_dropdowns(),
            dcc.Graph(id='variable-plot', style={'height': '800px', 'margin-top': '50px'}),
            html.Div(id='stats-table'),
            dcc.Loading(id="loading-1", type="default", children=html.Div(id="loading-output-1"))
        ])

    def create_dropdowns(self):
        return html.Div([
            html.Div([
                dcc.Dropdown(
                    id='zone-dropdown',
                    options=[{'label': 'All', 'value': 'All'}] + [{'label': zone, 'value': zone} for zone in self.data.keys()],
                    value=['All'], multi=True, placeholder="Select Zones"
                ),
                dcc.Dropdown(id='instant-dropdown', multi=True, placeholder="Select Instants", value=['All']),
            ], style={'width': '48%', 'display': 'inline-block'}),

            html.Div([
                dcc.Dropdown(id='x-variable', placeholder="Select X Variable"),
                dcc.Dropdown(id='y-variable', multi=True, placeholder="Select Y Variables"),
                dcc.Dropdown(id='z-variable', multi=True, placeholder="Select Z Variables (optional for 3D plot)", style={'z-index': '1'})
            ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
        ])

    def get_available_variables(self, selected_zones, selected_instants):
        selected_zones = self._get_selected_items(selected_zones, self.data.keys())
        selected_instants = self._get_selected_items(selected_instants, set().union(*[self.data[zone].keys() for zone in selected_zones]))

        variables = set()
        for zone in selected_zones:
            for instant in selected_instants:
                if instant in self.data[zone]:
                    variables.update(self.data[zone][instant].keys())

        return list(variables)

    def _get_selected_items(self, selected_items, all_items):
        if 'All' in selected_items:
            return list(all_items)
        return selected_items

    def update_instants(self, selected_zones):
        if not selected_zones:
            return []

        selected_zones = self._get_selected_items(selected_zones, self.data.keys())

        instants = set()
        for zone in selected_zones:
            instants.update(self.data[zone].keys())

        return [{'label': 'All', 'value': 'All'}] + [{'label': instant, 'value': instant} for instant in instants]

    def update_variable_dropdowns(self, selected_zones, selected_instants):
        if not selected_zones or not selected_instants:
            return [], [], []

        variables = self.get_available_variables(selected_zones, selected_instants)

        x_options = [{'label': var, 'value': var} for var in variables]
        y_z_options = [{'label': 'All', 'value': 'All'}] + x_options

        return x_options, y_z_options, y_z_options

    def update_graph_and_stats(self, selected_zones, selected_instants, x_var, y_vars, z_vars):
        start_time = time.time()
        fig = go.Figure()

        if not selected_zones or not selected_instants or not x_var or not y_vars:
            return fig, None

        selected_zones = self._get_selected_items(selected_zones, self.data.keys())
        selected_instants = self._get_selected_items(selected_instants, set().union(*[self.data[zone].keys() for zone in selected_zones]))

        available_vars = self.get_available_variables(selected_zones, selected_instants)
        if 'All' in y_vars:
            y_vars = [var for var in available_vars if var != x_var]
        if z_vars and 'All' in z_vars:
            z_vars = [var for var in available_vars if var != x_var and var not in y_vars]
        elif not z_vars:
            z_vars = []

        fig, stats_df = self.plot_data_and_compute_stats(selected_zones, selected_instants, x_var, y_vars, z_vars, fig)

        stats_table = self.create_stats_table(stats_df)

        end_time = time.time()
        print(f"Time taken to update graph: {end_time - start_time:.2f}s")

        return fig, stats_table

    def plot_data_and_compute_stats(self, selected_zones, selected_instants, x_var, y_vars, z_vars, fig):
        stats_data = defaultdict(list)
        is_3d = len(z_vars) > 0

        for zone in selected_zones:
            for instant in selected_instants:
                if instant in self.data[zone]:
                    x_data = self.data[zone][instant].get(x_var, [])
                    
                    if is_3d:
                        for y_var in y_vars:
                            y_data = self.data[zone][instant].get(y_var, [])
                            stats_data[y_var].extend(y_data)
                            for z_var in z_vars:
                                z_data = self.data[zone][instant].get(z_var, [])
                                stats_data[z_var].extend(z_data)
                                if len(x_data) == len(y_data) == len(z_data):
                                    fig.add_trace(go.Scatter3d(
                                        x=x_data, y=y_data, z=z_data,
                                        mode='markers',
                                        name=f"{zone}_{instant}_{y_var}_{z_var}"
                                    ))
                    else:
                        for y_var in y_vars:
                            y_data = self.data[zone][instant].get(y_var, [])
                            stats_data[y_var].extend(y_data)
                            if len(x_data) == len(y_data):
                                fig.add_trace(go.Scatter(
                                    x=x_data, y=y_data,
                                    mode='lines',
                                    name=f"{zone}_{instant}_{y_var}"
                                ))

        if is_3d:
            fig.update_layout(
                scene=dict(
                    xaxis_title=x_var,
                    yaxis_title=', '.join(y_vars),
                    zaxis_title=', '.join(z_vars)
                ),
                title='3D Variables Plot'
            )
        else:
            fig.update_layout(
                xaxis_title=x_var,
                yaxis_title=', '.join(y_vars),
                title='2D Variables Plot'
            )

        fig.update_layout(
            xaxis=dict(
                rangeslider=dict(visible=True),
                type="linear"
            ),
            legend=dict(
                title='Traces',
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.5)"
            ),
            showlegend=True
        )
        
        stats_df = pd.DataFrame(stats_data).describe()
        stats_df.insert(0, 'Stat', stats_df.index)

        return fig, stats_df

    def create_stats_table(self, stats_df):
        return dash_table.DataTable(
            columns=[{'name': col, 'id': col} for col in stats_df.columns],
            data=stats_df.to_dict('records'),
            style_table={'overflowX': 'scroll'}
        )

    def setup_callbacks(self):
        @self.app.callback(
            Output('instant-dropdown', 'options'),
            Input('zone-dropdown', 'value')
        )
        def update_instants_callback(selected_zones):
            return self.update_instants(selected_zones)

        @self.app.callback(
            [Output('x-variable', 'options'),
             Output('y-variable', 'options'),
             Output('z-variable', 'options')],
            [Input('zone-dropdown', 'value'),
             Input('instant-dropdown', 'value')]
        )
        def update_variable_dropdowns_callback(selected_zones, selected_instants):
            return self.update_variable_dropdowns(selected_zones, selected_instants)

        @self.app.callback(
            [Output('variable-plot', 'figure'),
             Output('stats-table', 'children')],
            [Input('zone-dropdown', 'value'),
             Input('instant-dropdown', 'value'),
             Input('x-variable', 'value'),
             Input('y-variable', 'value'),
             Input('z-variable', 'value')]
        )
        @self.cache.memoize(timeout=30)
        def update_graph_and_stats_callback(selected_zones, selected_instants, x_var, y_vars, z_vars):
            return self.update_graph_and_stats(selected_zones, selected_instants, x_var, y_vars, z_vars)

    def dash(self):
        self.app.layout = self.create_layout()
        self.setup_callbacks()
        self.app.run_server(debug=True, port=8051)

# Sample data structure generation
def generate_sample_data(n=1e3, num_zones=5):
    data = {}
    x = np.arange(0, n)
    for zone_id in range(num_zones):
        zone_name = f"Zone {zone_id+1}"
        data[zone_name] = {}
        for instant_id in range(10):
            instant_name = f"Instant {instant_id+1}"
            data[zone_name][instant_name] = {
                f'X': x,
                f'Y_{instant_id+1}': np.sin(x),
                f'Ybis_{instant_id+1}': np.cos(x),
                f'Z_{instant_id+1}': np.tan(x),
                f'Zbis_{instant_id+1}': np.sqrt(x),
            }
    return data

if __name__ == '__main__':
    data = generate_sample_data()
    plotter = Plotter(data)
    plotter.dash()
