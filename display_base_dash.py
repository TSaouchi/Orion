import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
from plotly_resampler import FigureResampler
from flask_caching import Cache
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import time
from scipy import stats

# Define constants
TIMEOUT = 30

# Sample data structure generation
def generate_sample_data(n=1e3, num_zones=500):
    """
    Generate sample data for the dashboard.
    
    Parameters:
        n: Number of data points per variable.
        num_zones: Number of zones and instants to create.
    
    Returns:
        data: Dictionary containing zones, instants, and variables.
    """
    data = {}
    val = np.arange(0, n)
    for i in range(num_zones):
        zone_name = f"zone_toto_and_long_name_{i}"
        instant_name = f"instant_toto_and_long_name_{i}"
        data[zone_name] = {instant_name: {}}
        data[zone_name][instant_name][f"var_{i}"] = np.sin(i * val) * np.cos(i * val)
        data[zone_name][instant_name]["TimeValue"] = i * val / 2.0
    return data

# Initialize data
data = generate_sample_data()

# Initialize Dash app and cache
app = dash.Dash(__name__)
cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory',
    'CACHE_DEFAULT_TIMEOUT': TIMEOUT
})

def create_layout():
    """
    Create the layout for the Dash app.
    
    Returns:
        A Dash HTML layout containing the dropdowns, graph, and stats table.
    """
    return html.Div([
        html.H1("Data Visualization Dashboard"),
        
        # Dropdowns for zones, instants, and variables
        html.Div([
            html.Div([
                dcc.Dropdown(id='zone-dropdown',
                             options=[{'label': 'All', 'value': 'All'}] + [{'label': zone, 'value': zone} for zone in data.keys()],
                             value=['All'], multi=True, placeholder="Select Zones"),
                dcc.Dropdown(id='instant-dropdown', multi=True, placeholder="Select Instants", value=['All']),
            ], style={'width': '48%', 'display': 'inline-block'}),

            html.Div([
                dcc.Dropdown(id='x-variable', placeholder="Select X Variable"),
                dcc.Dropdown(id='y-variable', multi=True, placeholder="Select Y Variables"),
                dcc.Dropdown(id='z-variable', multi=True, placeholder="Select Z Variables (optional for 3D plot)", style={'z-index': '1'})
            ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
        ]),

        # Graph for plotting variables
        dcc.Graph(id='variable-plot', style={'height': '800px', 'margin-top': '50px'}),

        # Table for displaying statistics
        html.Div(id='stats-table')
    ])

app.layout = create_layout()

# Helper function for collecting variables
def get_available_variables(selected_zones, selected_instants):
    """
    Get available variables based on selected zones and instants.
    
    Parameters:
        selected_zones: List of selected zones.
        selected_instants: List of selected instants.
    
    Returns:
        List of available variables.
    """
    if 'All' in selected_zones:
        selected_zones = list(data.keys())
    
    if 'All' in selected_instants:
        selected_instants = set()
        for zone in selected_zones:
            selected_instants.update(data[zone].keys())
    
    variables = set()
    for zone in selected_zones:
        for instant in selected_instants:
            if instant in data[zone]:
                variables.update(data[zone][instant].keys())
    
    return list(variables)

# Callbacks
@app.callback(
    Output('instant-dropdown', 'options'),
    Input('zone-dropdown', 'value')
)
def update_instants(selected_zones):
    """
    Update the list of instants based on selected zones.
    
    Parameters:
        selected_zones: List of selected zones from the dropdown.
    
    Returns:
        A list of options for the instants dropdown.
    """
    if not selected_zones:
        return []

    if 'All' in selected_zones:
        selected_zones = list(data.keys())

    instants = set()
    for zone in selected_zones:
        instants.update(data[zone].keys())

    return [{'label': 'All', 'value': 'All'}] + [{'label': instant, 'value': instant} for instant in instants]

@app.callback(
    [Output('x-variable', 'options'),
     Output('y-variable', 'options'),
     Output('z-variable', 'options')],
    [Input('zone-dropdown', 'value'),
     Input('instant-dropdown', 'value')]
)
def update_variable_dropdowns(selected_zones, selected_instants):
    """
    Update the variable dropdowns based on selected zones and instants.
    
    Parameters:
        selected_zones: List of selected zones.
        selected_instants: List of selected instants.
    
    Returns:
        Options for X, Y, and Z variable dropdowns.
    """
    if not selected_zones or not selected_instants:
        return [], [], []

    variables = get_available_variables(selected_zones, selected_instants)

    x_options = [{'label': var, 'value': var} for var in variables]
    y_z_options = [{'label': 'All', 'value': 'All'}] + x_options

    return x_options, y_z_options, y_z_options

@app.callback(
    [Output('variable-plot', 'figure'),
     Output('stats-table', 'children')],
    [Input('zone-dropdown', 'value'),
     Input('instant-dropdown', 'value'),
     Input('x-variable', 'value'),
     Input('y-variable', 'value'),
     Input('z-variable', 'value')]
)
@cache.memoize(timeout=TIMEOUT)
def update_graph_and_stats(selected_zones, selected_instants, x_var, y_vars, z_vars):
    """
    Update the graph and statistics table based on selected variables.
    
    Parameters:
        selected_zones: List of selected zones.
        selected_instants: List of selected instants.
        x_var: Selected X variable.
        y_vars: Selected Y variables.
        z_vars: Selected Z variables (optional for 3D plotting).
    
    Returns:
        Updated Plotly figure and statistics table.
    """
    start_time = time.time()
    fig = FigureResampler(go.Figure())

    if not selected_zones or not selected_instants or not x_var or not y_vars:
        return fig, None

    # Handle 'All' options for zones and instants
    if 'All' in selected_zones:
        selected_zones = list(data.keys())
    if 'All' in selected_instants:
        selected_instants = set()
        for zone in selected_zones:
            selected_instants.update(data[zone].keys())
    selected_instants = list(selected_instants)

    # Handle 'All' option for Y and Z variables
    available_vars = get_available_variables(selected_zones, selected_instants)
    if 'All' in y_vars:
        y_vars = [var for var in available_vars if var != x_var]
    if z_vars and 'All' in z_vars:
        z_vars = [var for var in available_vars if var != x_var and var not in y_vars]
    elif not z_vars:
        z_vars = []  # Ensure z_vars is a list even if no Z variables are selected

    # Collect data and create traces
    fig, stats_df = plot_data_and_compute_stats(selected_zones, selected_instants, x_var, y_vars, z_vars, fig)

    # Create statistics table
    stats_table = create_stats_table(stats_df)

    end_time = time.time()
    print(f"Time taken to update graph: {end_time - start_time:.2f}s")

    return fig, stats_table

def plot_data_and_compute_stats(selected_zones, selected_instants, x_var, y_vars, z_vars, fig):
    """
    Plot data for selected variables and compute statistics.
    
    Returns:
        Updated Plotly figure and Pandas DataFrame with statistics.
    """
    stats_data = {var: [] for var in y_vars + z_vars}
    is_3d = len(z_vars) > 0

    # Iterate through the selected zones and instants to plot and compute statistics
    for zone in selected_zones:
        for instant in selected_instants:
            if instant in data[zone]:
                x_data = data[zone][instant].get(x_var, [])
                
                if is_3d:
                    for y_var in y_vars:
                        y_data = data[zone][instant].get(y_var, [])
                        stats_data[y_var].extend(y_data)
                        
                        for z_var in z_vars:
                            z_data = data[zone][instant].get(z_var, [])
                            stats_data[z_var].extend(z_data)
                            if len(x_data) == len(y_data) == len(z_data):
                                fig.add_trace(go.Scatter3d(
                                    x=x_data, y=y_data, z=z_data,
                                    mode='markers',
                                    name=f"{zone}_{instant}_{y_var}_{z_var}"
                                ))
                else:
                    for y_var in y_vars:
                        y_data = data[zone][instant].get(y_var, [])
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

    # Compute stats using stats_data dictionary
    stats_df = pd.DataFrame(stats_data).describe()
    stats_df.insert(0, 'Stat', stats_df.index)  # Add a 'Stat' column to label rows

    return fig, stats_df

def create_stats_table(stats_df):
    """
    Create a Dash table from the statistics DataFrame.
    
    Parameters:
        stats_df: DataFrame containing statistical data.
    
    Returns:
        Dash table component.
    """
    return dash_table.DataTable(
        columns=[{'name': col, 'id': col} for col in stats_df.columns],
        data=stats_df.to_dict('records'),
        style_table={'overflowX': 'scroll'}
    )

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)