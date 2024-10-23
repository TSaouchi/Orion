import numpy as np
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
from flask_caching import Cache
import plotly.graph_objs as go
import pandas as pd
import time

from Utils import compute_stats

class Plotter:
    def __init__(self, base):
        self.base = base
        self.app = dash.Dash(__name__)
        self.cache = Cache(self.app.server, config={
            'CACHE_TYPE': 'filesystem',
            'CACHE_DIR': 'cache-directory',
            'CACHE_DEFAULT_TIMEOUT': 120
        })

    def create_layout(self):
        return html.Div([
            # Title
            html.H1("Data Visualization Dashboard"),
            # DropDown menus
            self.create_dropdowns(),
            # Graph view zone
            dcc.Graph(id='variable-plot', 
                      style={'height': '800px', 'margin-top': '50px'}),
            # Stats Table zone
            html.Div(id='stats-table'),
            # Looding bar (not working!!)
            dcc.Loading(id="loading-1", 
                        type="default", children=html.Div(id="loading-output-1"))
        ])

    def create_dropdowns(self):
        return html.Div([
            html.Div([
                dcc.Dropdown(
                    id='zone-dropdown',
                    options=[{'label': 'All', 'value': 'All'}] + \
                        [{'label': zone, 'value': zone} for zone in self.base.keys()],
                    value=['All'], 
                    multi=True, 
                    placeholder="Select Zones"
                ),
                
                dcc.Dropdown(
                    id='instant-dropdown', 
                    multi=True, 
                    placeholder="Select Instants", 
                    value=['All']),
            ], style={'width': '48%', 'display': 'inline-block'}),

            html.Div([
                dcc.Dropdown(
                    id='x-variable', 
                    placeholder="Select X Variable"),
                dcc.Dropdown(
                    id='y-variable',
                    multi=True, 
                    placeholder="Select Y Variables"),
                dcc.Dropdown(
                    id='z-variable', 
                    multi=True, 
                    placeholder="Select Z Variables (optional for 3D plot)", style={'z-index': '1'})
            ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
        ])

    def get_available_variables(self, selected_zones, selected_instants):
        selected_zones = self._get_selected_items(selected_zones, 
                                                  self.base.keys())
        selected_instants = self._get_selected_items(selected_instants, 
                                                     set().union(*
                                                                 [self.base[zone].keys() for zone in selected_zones]
                                                                 )
                                                     )
        variables = set()
        for zone in selected_zones:
            for instant in selected_instants:
                if instant in self.base[zone].keys():
                    variables.update(self.base[zone][instant].keys())

        return sorted(list(variables))

    def _get_selected_items(self, selected_items, all_items):
        return list(all_items) if 'All' in selected_items else selected_items

    def update_instants(self, selected_zones):
        if not selected_zones:
            return []

        selected_zones = self._get_selected_items(selected_zones, self.base.keys())
        instants = sorted([*set().union(*[self.base[zone].keys() for zone in selected_zones])])

        return [{'label': 'All', 'value': 'All'}] + [{'label': instant, 'value': instant} for instant in instants]

    def update_variable_dropdowns(self, selected_zones, selected_instants):
        if not selected_zones or not selected_instants:
            return [], [], []

        variables = self.get_available_variables(selected_zones, selected_instants)

        x_options = [{'label': var, 'value': var} for var in variables]
        y_z_options = [{'label': 'All', 'value': 'All'}] + x_options

        return x_options, y_z_options, y_z_options

    def aggregate_data(self, selected_zones, selected_instants, x_var, y_vars, z_vars):
        # Ensure y_vars and z_vars are not None
        y_vars = y_vars or []
        z_vars = z_vars or []
        
        aggregated_set = set()

        for zone in selected_zones:
            for instant in selected_instants:
                if instant in self.base[zone].keys():
                    # Collect X, Y, and Z data only once
                    if (x_var, zone, instant) not in aggregated_set:
                        aggregated_set.add((x_var, zone, instant))

                    for var_group, vars_list in zip([y_vars, z_vars], [y_vars, z_vars]):
                        for var in vars_list:
                            if (var, zone, instant) not in aggregated_set:
                                aggregated_set.add((var, zone, instant))

        return aggregated_set

    def update_graph_and_stats(self, selected_zones, selected_instants, 
                               x_var, y_vars, z_vars):
        start_time = time.time()

        if not selected_zones or not selected_instants or not x_var or not y_vars:
            return go.Figure(), None

        if 'All' in y_vars: 
            y_vars = list(
                set(self.get_available_variables(
                    selected_zones, 
                    selected_instants)
                    ) - set(x_var))
            
        if z_vars is not None and 'All' in z_vars:
            z_vars = list(
                set(self.get_available_variables(
                    selected_zones, 
                    selected_instants)
                    ) - set(x_var))
            
        selected_zones = self._get_selected_items(selected_zones, self.base.keys())
        selected_instants = self._get_selected_items(
            selected_instants, 
            set().union(*[self.base[zone].keys() for zone in selected_zones])
            )

        aggregated_set = self.aggregate_data(selected_zones, selected_instants, 
                                             x_var, y_vars, z_vars)
        is_3d = z_vars is not None and len(z_vars) > 0

        fig = self.create_figure(aggregated_set, x_var, y_vars or [], 
                                 z_vars or [], is_3d)

        stats_df = pd.DataFrame()
        for variable, zone, instant in aggregated_set:
            if variable not in self.base[zone][instant].keys(): continue
            stats_names, variable_stats = \
                compute_stats(self.base[zone][instant][variable].data)
            stats_df.insert(0, f"{zone}/{instant}/{variable}", variable_stats)
        stats_df.insert(0, 'Stat', stats_names)

        end_time = time.time()
        print(f"Time taken to update graph: {end_time - start_time:.2f}s")

        return fig, self.create_stats_table(stats_df)
    
    def __get_data(self, variable, zone, instant):
        return self.base[zone][instant].get(variable, []).data if hasattr(self.base[zone][instant].get(variable, []), 'data') else []

    def create_figure(self, aggregated_set, x_var, y_vars, 
                      z_vars, is_3d):
        fig = go.Figure()

        if is_3d:
            for y_var in y_vars:
                for variable, zone, instant in aggregated_set:
                    if variable != y_var: continue
                    for z_var in z_vars:
                        fig.add_trace(go.Scatter3d(
                            x=self.__get_data(x_var, zone, instant),
                            y=self.__get_data(y_var, zone, instant),
                            z=self.__get_data(z_var, zone, instant),
                            mode='markers',
                            name=f"{zone}_{instant}_{y_var}_{z_var}"
                        ))

                fig.update_layout(
                    scene=dict(
                        xaxis_title=x_var,
                        yaxis_title=', '.join(y_vars),
                        zaxis_title=', '.join(z_vars)
                    ),
                    title='3D Variables Plot'
                )
        else:
            for y_var in y_vars:
                for variable, zone, instant in aggregated_set:
                    if variable != y_var: continue
                    fig.add_trace(go.Scatter(
                        x=self.__get_data(x_var, zone, instant),
                        y=self.__get_data(y_var, zone, instant),
                        mode='lines',
                        name=f"{zone}_{instant}_{y_var}"
                    ))

                    fig.update_layout(
                        xaxis_title=x_var,
                        yaxis_title=', '.join(y_vars),
                        title='2D Variables Plot'
                    )

        fig.update_layout(
            xaxis=dict(rangeslider=dict(visible=True), type="linear"),
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

        return fig

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
        @self.cache.memoize(timeout=40)
        def update_graph_callback(selected_zones, selected_instants, x_var, y_vars, z_vars):
            return self.update_graph_and_stats(selected_zones, selected_instants, x_var, y_vars, z_vars)

    def run(self):
        self.app.layout = self.create_layout()
        self.setup_callbacks()
        self.app.run_server(debug=True)


if __name__ == '__main__':
    
    import Core as Orion
    import dask.array as da
    def generate_sample_data():
        base = Orion.Base()
        nzone = 5
        ninstant = 5
        n = 10
        zones = [f"Zone_{i}" for i in range(nzone)]
        instants = [f"Instant_{i}" for i in range(ninstant)]
        base.init(zones, instants)
        
        var1 = [f"var_{i}" for i in range(0, 5)]
        var2 = [f"var_{i}" for i in range(50, 60)]
        var3 = [f"var_{i}" for i in range(60, 70)]
        
        for zone in zones[0:2]:
            for instant in instants:
                for var in var1:
                    base[zone][instant].add_variable(var, 
                                                     da.random.random(n, 1)
                                                     )
        for zone in zones[2:3]:
            for instant in instants:
                for var in var2:
                    base[zone][instant].add_variable(var, 
                                                     da.random.random(n, 1)
                                                     )
        for zone in zones[3:5]:
            for instant in instants:
                for var in var3:
                    base[zone][instant].add_variable(var, 
                                                     da.random.random(n, 1)
                                                     )
        return base
    data = generate_sample_data()
    
    plotter = Plotter(data)
    plotter.run()
