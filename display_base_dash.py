### Path and Char manipulation
import os

### Math
import numpy as np
import copy
import scipy as spy

import Core as Orion
from DataProcessor import Processor
from Utils import *

Reader = Orion.Reader
def tmp_reader_BHG(base_blueprint, Reader):
    base = Orion.Base()
    for zone, config in base_blueprint.items():
        base.add_zone(zone)
        files_dir = list(Reader(config["path"], patterns=config["dir_name_pattern"]
                            )._Reader__generate_file_list(config["path"], 
                                                          patterns=config["dir_name_pattern"]
                                                          ))
        for dir in files_dir:
            try:
                files = list(Reader(os.path.join(config["path"], dir), 
                                    patterns=config["file_name_pattern"]
                                    )._Reader__generate_file_list(
                                        os.path.join(config["path"], dir), 
                                        patterns=config["file_name_pattern"]))
            except: continue
            
            instant_names = [name.split('.')[0] for name in files]
            files = [os.path.join(config["path"], dir, file) for file in files]    

            for file, instant_name in zip(files, instant_names):
                base[zone].add_instant(instant_name)
                # Read each file in a dictionary
                file_data = {}
                spy.io.loadmat(file, mdict = file_data)
                
                # Base attributes
                file_attr = file_data['File_Header']
                for attr in file_attr.dtype.names:
                    base.set_attribute(attr, file_attr[attr][0][0][0])
                
                channel_number = int(base.get_attribute('NumberOfChannels'))
                variable_names = [file_data[f'Channel_{_}_Header']['SignalName'][0][0][0]  for _ in range(1, channel_number)]
                
                for nvariable, variable in enumerate(variable_names, start=1):
                    base[zone][instant_name].add_variable(variable, 
                                                        [i[0] for i in file_data[f'Channel_{nvariable}_Data']])
    return Orion.SharedMethods().variable_mapping_cgns(base)

if __name__ == "__main__":
    # ========================= Cases configuration ============================
    # ============================== Read inputs ===============================
    direction = "From_1_to_2"
    base_blueprint = {
        "RT" : {
            "path" : r"C:\__sandBox__\Data\SARC_Gen2_HydraulicTest_Hydac\EWR12857-145",
            "dir_name_pattern" : "Resp*",
            "file_name_pattern" : "Valve_Hydac_SN_15_From_2_to_1*"
        }
    }
    # Inputs
    base = tmp_reader_BHG(base_blueprint, Reader)
    base = Processor(base).normalization()
    
    import dask.array as da
    # Sample data structure
    data = {}
    for i in range(500):
        data[f"zone_toto_and_long_name_{i}"] = {}
        data[f"zone_toto_and_long_name_{i}"][f"instant_toto_and_long_name_{i}"] = {}
        for var, val in base[0][0].items():
            data[f"zone_toto_and_long_name_{i}"][f"instant_toto_and_long_name_{i}"][var] = da.concatenate([val, val + np.max(val), val + 2*np.max(val)])
    
    from SharedMethods import SharedMethods
    var_base = SharedMethods().variables_location(base)
    for zone in range(int(10)):
        base.add_zone(f'toto{zone}')
        for instant in range(int(10)):
            base[f'toto{zone}'].add_instant(f"instant{instant}")
            for var in var_base:
                base[f'toto{zone}'][f"instant{instant}"].add_variable(var, val)
    
    import time
    start_time = time.time()
    base.compute('TOTO = TimeValue * 5')
    elapsed_time = time.time() - start_time  # Calculate elapsed time
    print(f"update_instants execution time: {elapsed_time:.4f} seconds")  # Log the time
        
    import dash
    from dash import dcc, html
    from plotly_resampler import FigureResampler
    from flask_caching import Cache
    from dash.dependencies import Input, Output
    import plotly.graph_objs as go
    import numpy as np
    import time
    import dask.array as da

    # Sample data structure
    data = {}
    for i in range(500):
        data[f"zone_toto_and_long_name_{i}"] = {}
        data[f"zone_toto_and_long_name_{i}"][f"instant_toto_and_long_name_{i}"] = {}
        for var, val in base[0][0].items():
            data[f"zone_toto_and_long_name_{i}"][f"instant_toto_and_long_name_{i}"][var] = da.concatenate([val, val + np.max(val), val + 2*np.max(val)])

    app = dash.Dash(__name__)
    TIMEOUT = 30
    cache = Cache(app.server, config={
        'CACHE_TYPE': 'filesystem',
        'CACHE_DIR': 'cache-directory',
        'CACHE_DEFAULT_TIMEOUT': TIMEOUT
    })

    app.layout = html.Div([
        html.H1("Data Visualization Dashboard"),
        
        # Dropdown for Zones (allow multiple selections and Select All)
        dcc.Dropdown(
            id='zone-dropdown',
            options=[{'label': 'All', 'value': 'All'}] + [{'label': zone, 'value': zone} for zone in data.keys()],
            value=['All'],  # Default to "All" selected
            multi=True,  # Allow multiple zones to be selected
            placeholder="Select Zones"
        ),
        
        # Dropdown for Instants (dynamic based on selected zones, allow Select All)
        dcc.Dropdown(
            id='instant-dropdown',
            multi=True,  # Allow multiple instants to be selected
            placeholder="Select Instants",
            value=['All']  # Default to "All" selected
        ),
        
        # Dropdown for Variables (dynamic based on selected zones and instants, allow Select All)
        dcc.Dropdown(
            id='variable-dropdown',
            multi=True,  # Allow multiple variables to be selected
            placeholder="Select Variables",
            value=[]  # Default nothing selected
        ),
        
        # Graph to plot the variables
        dcc.Graph(id='variable-plot')
    ])

    # Callback to update instants based on selected zones
    @app.callback(
        Output('instant-dropdown', 'options'),
        Output('instant-dropdown', 'value'),
        Input('zone-dropdown', 'value')
    )
    def update_instants(selected_zones):
        start_time = time.time()  # Start timing
        if not selected_zones:
            return [], []  # If no zones are selected, return empty lists
        
        if 'All' in selected_zones:
            selected_zones = list(data.keys())  # Select all zones if "All" is selected
        
        instants = set()
        for zone in selected_zones:
            instants.update(data[zone].keys())
        
        instant_options = [{'label': 'All', 'value': 'All'}] + [{'label': instant, 'value': instant} for instant in instants]
        
        elapsed_time = time.time() - start_time  # Calculate elapsed time
        print(f"update_instants execution time: {elapsed_time:.4f} seconds")  # Log the time
        
        return instant_options, ['All']  # Default to "All" selected

    # Callback to update variables based on selected zones and instants
    @app.callback(
        Output('variable-dropdown', 'options'),
        Output('variable-dropdown', 'value'),
        Input('zone-dropdown', 'value'),
        Input('instant-dropdown', 'value'),
        Input('variable-dropdown', 'value')  # Added to capture currently selected variables
    )
    def update_variables(selected_zones, selected_instants, selected_variables):
        start_time = time.time()  # Start timing
        if not selected_zones or not selected_instants:
            return [], []  # If no zones or instants are selected, return empty lists
        
        if 'All' in selected_zones:
            selected_zones = list(data.keys())  # Select all zones if "All" is selected
        
        if 'All' in selected_instants:
            selected_instants = set()
            for zone in selected_zones:
                selected_instants.update(data[zone].keys())  # Select all instants if "All" is selected

        variables = set()
        for zone in selected_zones:
            for instant in selected_instants:
                if instant in data[zone]:
                    variables.update(data[zone][instant].keys())
        
        variable_options = [{'label': 'All', 'value': 'All'}] + [{'label': var, 'value': var} for var in variables]
        
        # Preserve previously selected variables if they are still available
        current_selected_variables = [var for var in variable_options if var['value'] in selected_variables]
        selected_value = [var['value'] for var in current_selected_variables]
        
        elapsed_time = time.time() - start_time  # Calculate elapsed time
        print(f"update_Variables execution time: {elapsed_time:.4f} seconds")  # Log the time
        
        return variable_options, selected_value  # Return available variables and preserve selected variables

    # Callback to plot the selected variables
    @app.callback(
        Output('variable-plot', 'figure'),
        Input('zone-dropdown', 'value'),
        Input('instant-dropdown', 'value'),
        Input('variable-dropdown', 'value')
    )
    @cache.memoize(timeout=TIMEOUT)
    def plot_variables(selected_zones, selected_instants, selected_variables):
        start_time = time.time()  # Start timing
        fig = FigureResampler(go.Figure())

        if not selected_zones or not selected_instants:
            fig.update_layout(
                title='No Data Selected',
                xaxis={'title': 'Time'},
                yaxis={'title': 'Value'}
            )
            return fig

        if 'All' in selected_zones:
            selected_zones = list(data.keys())  # Select all zones if "All" is selected

        if 'All' in selected_instants:
            # If instants are set to "All," get all instants for the selected zones
            selected_instants = set()
            for zone in selected_zones:
                selected_instants.update(data[zone].keys())  # Select all instants if "All" is selected

        if 'All' in selected_variables:
            # If variables are set to "All," get all variables for the selected zones and instants
            variables = set()
            for zone in selected_zones:
                for instant in selected_instants:
                    if instant in data[zone]:  # Ensure instant exists in zone
                        variables.update(data[zone][instant].keys())
            selected_variables = list(variables)  # Update selected_variables to include all variables for the selected zones and instants

        for zone in selected_zones:
            for instant in selected_instants:
                if instant in data[zone]:  # Ensure instant exists in zone
                    time_values = data[zone][instant]['TimeValue']
                    for var in selected_variables:
                        if var in data[zone][instant]:  # Ensure variable exists in instant
                            trace_name = f"{zone}_{instant}_{var}"
                            fig.add_trace(go.Scatter(
                                x=time_values,
                                y=data[zone][instant][var],
                                mode='lines',
                                name=trace_name  # Use custom name for legend
                            ))

        fig.update_layout(
            title='Variables Plot',
            xaxis={'title': 'Time'},
            yaxis={'title': 'Value'},
            legend=dict(
                title='Variables',
                itemclick='toggle',  # Allow clickable legend items
                itemdoubleclick='toggleothers'  # Allow double-click to isolate
            )
        )

        fig.update_layout(
            xaxis=dict(
                rangeslider=dict(
                    visible=True
                ),
            )
        )
        elapsed_time = time.time() - start_time  # Calculate elapsed time
        print(f"Plot variables execution time: {elapsed_time:.4f} seconds")  # Log the time
        
        return fig

    app.run_server(debug=True)