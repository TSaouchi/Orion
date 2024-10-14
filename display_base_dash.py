import dash
from dash import dcc, html
from plotly_resampler import FigureResampler
from flask_caching import Cache
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np
import time

# Sample data structure (unchanged)
data = {}
n = 1e3
val = np.arange(0, n)
for i in range(500):
    data[f"zone_toto_and_long_name_{i}"] = {}
    data[f"zone_toto_and_long_name_{i}"][f"instant_toto_and_long_name_{i}"] = {}
    data[f"zone_toto_and_long_name_{i}"][f"instant_toto_and_long_name_{i}"][f"var_{i}"] =  np.sin(i*val)*np.cos(i*val)
    data[f"zone_toto_and_long_name_{i}"][f"instant_toto_and_long_name_{i}"]["TimeValue"] = i*val/2.0

app = dash.Dash(__name__)
TIMEOUT = 30
cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory',
    'CACHE_DEFAULT_TIMEOUT': TIMEOUT
})

app.layout = html.Div([
    html.H1("Data Visualization Dashboard"),
    
    dcc.Dropdown(id='zone-dropdown', options=[{'label': 'All', 'value': 'All'}] + [{'label': zone, 'value': zone} for zone in data.keys()], value=['All'], multi=True, placeholder="Select Zones"),
    dcc.Dropdown(id='instant-dropdown', multi=True, placeholder="Select Instants", value=['All']),
    dcc.Dropdown(id='x-variable', placeholder="Select X Variable"),
    dcc.Dropdown(id='y-variable', multi=True, placeholder="Select Y Variables"),
    dcc.Dropdown(id='z-variable', multi=True, placeholder="Select Z Variables (optional for 3D plot)"),
    
    dcc.Graph(id='variable-plot')
])

@app.callback(
    Output('instant-dropdown', 'options'),
    Output('instant-dropdown', 'value'),
    Input('zone-dropdown', 'value')
)
def update_instants(selected_zones):
    if not selected_zones:
        return [], []
    
    if 'All' in selected_zones:
        selected_zones = list(data.keys())
    
    instants = set()
    for zone in selected_zones:
        instants.update(data[zone].keys())
    
    instant_options = [{'label': 'All', 'value': 'All'}] + [{'label': instant, 'value': instant} for instant in instants]
    
    return instant_options, ['All']

@app.callback(
    [Output('x-variable', 'options'),
     Output('y-variable', 'options'),
     Output('z-variable', 'options')],
    [Input('zone-dropdown', 'value'),
     Input('instant-dropdown', 'value')]
)
def update_variable_dropdowns(selected_zones, selected_instants):
    if not selected_zones or not selected_instants:
        return [], [], []
    
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
    
    x_options = [{'label': var, 'value': var} for var in variables]
    y_z_options = [{'label': 'All', 'value': 'All'}] + x_options
    
    return x_options, y_z_options, y_z_options

@app.callback(
    Output('variable-plot', 'figure'),
    [Input('zone-dropdown', 'value'),
     Input('instant-dropdown', 'value'),
     Input('x-variable', 'value'),
     Input('y-variable', 'value'),
     Input('z-variable', 'value')]
)
@cache.memoize(timeout=TIMEOUT)
def plot_variables(selected_zones, selected_instants, x_var, y_vars, z_vars):
    start_time = time.time()
    fig = FigureResampler(go.Figure())

    if not selected_zones or not selected_instants or not x_var or not y_vars:
        fig.update_layout(title='Insufficient Data Selected')
        return fig

    if 'All' in selected_zones:
        selected_zones = list(data.keys())

    if 'All' in selected_instants:
        selected_instants = set()
        for zone in selected_zones:
            selected_instants.update(data[zone].keys())

    if 'All' in y_vars:
        y_vars = [var for var in data[list(data.keys())[0]][list(data[list(data.keys())[0]].keys())[0]].keys() if var != x_var]

    if z_vars and 'All' in z_vars:
        z_vars = [var for var in data[list(data.keys())[0]][list(data[list(data.keys())[0]].keys())[0]].keys() if var != x_var and var not in y_vars]

    is_3d = z_vars and len(z_vars) > 0

    for zone in selected_zones:
        for instant in selected_instants:
            if instant in data[zone]:
                x_data = data[zone][instant].get(x_var, [])
                
                for y_var in y_vars:
                    y_data = data[zone][instant].get(y_var, [])
                    
                    if is_3d:
                        for z_var in z_vars:
                            z_data = data[zone][instant].get(z_var, [])
                            if len(x_data) == len(y_data) == len(z_data):
                                fig.add_trace(go.Scatter3d(
                                    x=x_data, y=y_data, z=z_data,
                                    mode='markers',
                                    name=f"{zone}_{instant}_{y_var}_{z_var}"
                                ))
                    else:
                        if len(x_data) == len(y_data):
                            fig.add_trace(go.Scatter(
                                x=x_data, y=y_data,
                                mode='lines+markers',
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

    # Add rangeslider
    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(visible=True),
            type="linear"
        )
    )

    # Add legend
    fig.update_layout(
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

    elapsed_time = time.time() - start_time
    print(f"Plot variables execution time: {elapsed_time:.4f} seconds")
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)