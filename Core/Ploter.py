# Path and Char manipulation
import os

# Orion
import Core as Orion
from SharedMethods import SharedMethods

# Message mode
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['VTK_SILENCE_GET_VOID_POINTER_WARNINGS'] = '1'

class Plot(SharedMethods):
    """
    A class for grouping plotting methods.
    """
    def __init__(self, path, dir_name_tag = "output", files_name_tag = "output_result"):
        """
        Initialize the Plot class with path, directory name, and file name tags.

        Parameters
        ----------
        path : str
            The path where the output files will be saved.
        dir_name_tag : str, optional
            The directory name for saving the files (default is "output").
        files_name_tag : str, optional
            The base name of the output files (default is "output_result").

        Example
        -------
        >>> plot = Plot("/path/to/save", dir_name_tag="results", files_name_tag="plot_results")
        """
        self.dir_name_tag = dir_name_tag
        self.files_name_tag = files_name_tag
        self.path = path

    def cartesian_plot_to_html(self, plot_dictionary, auto_open = False):
        """
        Plot scatter points using Plotly in Cartesian coordinates.

        Parameters
        ----------
        plot_dictionary : dict
            A dictionary containing data for plotting. The dictionary should contain keys
            for 'x_axis_title', 'y_axis_title', and optionally 'z_axis_title'. Each key
            corresponds to a dictionary with 'values', 'markers', 'lines', 'sizes', 'scale' (2D plots), <axis>lim and 'legend'.
        auto_open : bool, optional
            If True, automatically open the plot in a browser (default is False).

        Notes
        -----
        - `plot_dictionary` should have the following structure:
        
        Example
        -------
        
        .. code-block:: python
        
            plot_dictionary = {
                'Coordinate X': {
                    'values': [x_values_1, x_values_2],
                    'markers': ['markers', 'lines'],
                    'sizes': [1, 5],
                    'legend': ['Data 1', 'Data 2'],
                    'xlim' : (None, 10),
                    'ylim' : (None, 10),
                    'secondary_y' : True,
                    'legend_secondary' : ['toto2'],
                },
                'Coordinate Y': {
                    'values': [y_values_1, y_values_2],
                }
            }
        
        Example Usage
        -------------
        >>> plot.cartesian_plot_to_html(plot_dictionary)
        # If one need to plot 2 left plots and one right plot use 
        >>> plot_dictionary = {
            'Time (s) - Shifted to start at zero' : {
                'values' : [base[0][0][0].data, base[0][0][0].data],
                'markers' : 2*['lines'],
                'legend' : 2*['toto1'],
                'secondary_y' : True,
                'legend_secondary' : ['toto2'],
            },
            'Displacement/BMW maximum dispalcement' : {
                'values' : [base[0][0][1].data, base[0][0][3].data]
            },
            'Displacement/BMW maximum ' : {
                'values' : [base[0][0][4].data,]
            }
        }
        """
        try:
            import plotly.graph_objects as go
            from plotly.offline import plot
        except ImportError as e:
            print(e)
            return

        self.print_text("info", "\nPlotting cartesian plot...")

        #:Get axis titles
        x_axis_title = list(plot_dictionary.keys())[0]
        y_axis_title = list(plot_dictionary.keys())[1]
        z_axis_title = list(plot_dictionary.keys())[2] if len(plot_dictionary) == 3 else None

        x_axis = plot_dictionary[x_axis_title]
        y_axis = plot_dictionary[y_axis_title]

        if z_axis_title is not None and x_axis.get('secondary_y', None) is not None:
            from plotly.subplots import make_subplots
            fig = make_subplots(specs=[[{"secondary_y": True}]])
        else:
            from plotly.subplots import make_subplots
            fig = make_subplots(specs=[[{"secondary_y": False}]])
            
        #:Plotting for 2D
        if z_axis_title is None:
            self.__2D_plot(go, fig, x_axis_title, y_axis_title, x_axis, y_axis)
        
        #:Plotting for 2D with secondary y
        elif z_axis_title is not None and x_axis.get('secondary_y', None) is not None:
            self.__2D_plot(go, fig, x_axis_title, y_axis_title, x_axis, y_axis)
            
            secondary_y = True
            z_axis = plot_dictionary[z_axis_title]
            self.__2D_plot(go, fig, x_axis_title, z_axis_title, x_axis, z_axis, 
                           secondary_y = secondary_y)
            fig.update_layout(
                    xaxis_title = x_axis_title,
                    yaxis_title = y_axis_title)
            fig.update_yaxes(title_text=z_axis_title, 
                             secondary_y = secondary_y)
            
        #:Plotting for 3D
        else:
            z_axis = plot_dictionary[z_axis_title]
            self.__3D_plot(go, fig, x_axis_title, y_axis_title, z_axis_title, 
                           x_axis, y_axis, z_axis)
                 
        export_path = self.export_path_check()
        output_path = os.path.join(export_path, self.files_name_tag)
        self.print_text("check", f"\n\tHTML Plot produced in : {output_path}")
        plot(fig, filename = output_path, auto_open = auto_open)

    def __3D_plot(self, go, fig, x_axis_title, y_axis_title, z_axis_title, x_axis, y_axis, z_axis):
        for n, (x, y, z) in enumerate(zip(x_axis['values'], y_axis['values'], z_axis['values'])):
            mode = x_axis.get('markers', ['markers'])[n]
            if mode == 'lines':
                fig.add_trace(go.Scatter3d(
                        x = x,
                        y = y,
                        z = z,
                        mode = 'lines',
                        name = x_axis['legend'][n] if 'legend' in x_axis else None,
                        line = dict(width = x_axis['sizes'][n] if 'sizes' in x_axis else 2)
                    ))
            elif mode == 'markers':
                fig.add_trace(go.Scatter3d(
                        x = x,
                        y = y,
                        z = z,
                        mode = 'markers',
                        name = x_axis['legend'][n] if 'legend' in x_axis else None,
                        marker = dict(size = x_axis['sizes'][n] if 'sizes' in x_axis else 10)
                    ))
            else:
                fig.add_trace(go.Scatter3d(
                        x = x,
                        y = y,
                        z = z,
                        mode = 'markers+lines',
                        name = x_axis['legend'][n] if 'legend' in x_axis else None,
                        marker = dict(size = x_axis['sizes'][n] if 'sizes' in x_axis else 10),
                        line = dict(width = x_axis['sizes'][n] if 'sizes' in x_axis else 2)
                    ))
            
        if 'scale' in x_axis.keys():
            if x_axis["scale"] in "loglog":
                fig.update_layout(
                    xaxis_title = x_axis_title,
                    yaxis_title = y_axis_title,
                    zaxis_title = z_axis_title,
                    xaxis=dict(type="log", exponentformat="e"),  
                    yaxis=dict(type="log", exponentformat="e"),  
                    zaxis=dict(type="log", exponentformat="e")  
                )
            elif x_axis["scale"] in "logx":
                fig.update_layout(
                    xaxis_title = x_axis_title,
                    yaxis_title = y_axis_title,
                    zaxis_title = z_axis_title,
                    xaxis=dict(type="log", exponentformat="e"),  
                )
            elif x_axis["scale"] in "logy":
                fig.update_layout(
                    xaxis_title = x_axis_title,
                    yaxis_title = y_axis_title,
                    zaxis_title = z_axis_title,
                    yaxis=dict(type="log", exponentformat="e")  
                )                
            elif x_axis["scale"] in "logz":
                fig.update_layout(
                    xaxis_title = x_axis_title,
                    yaxis_title = y_axis_title,
                    zaxis_title = z_axis_title,
                    zaxis=dict(type="log", exponentformat="e")  
                )                
        else:
            fig.update_layout(
                scene = dict(
                    xaxis_title = x_axis_title,
                    yaxis_title = y_axis_title,
                    zaxis_title = z_axis_title
                    )
                )
            
        if "xlim" in x_axis.keys():
             fig.update_xaxes(range = x_axis["xlim"])
        if "ylim" in x_axis.keys():
             fig.update_yaxes(range = x_axis["ylim"])
        if "zlim" in x_axis.keys():
             fig.update_zaxes(range = x_axis["zlim"])

    def __2D_plot(self, go, fig, x_axis_title, y_axis_title, x_axis, y_axis, 
                  secondary_y = False):
        for n, (x, y) in enumerate(zip(x_axis['values'], y_axis['values'])):
            mode = x_axis.get('markers', ['markers'])[n]
            if secondary_y:
                name = x_axis['legend_secondary'][n] if 'legend_secondary' in x_axis else None
            else:
                name = x_axis['legend'][n] if 'legend' in x_axis else None
            if mode == 'lines':
                fig.add_trace(go.Scatter(
                            x = x,
                            y = y,
                            mode = 'lines',
                            name = name,
                            line = dict(width = x_axis['sizes'][n] if 'sizes' in x_axis else 2)
                        ),
                        secondary_y = secondary_y
                        )
            elif mode == 'markers':
                fig.add_trace(go.Scatter(
                            x = x,
                            y = y,
                            mode = 'markers',
                            name = name,
                            marker = dict(size = x_axis['sizes'][n] if 'sizes' in x_axis else 10)
                        ),
                        secondary_y = secondary_y
                              )
            else:
                fig.add_trace(go.Scatter(
                            x = x,
                            y = y,
                            mode = 'markers+lines',
                            name = name,
                            marker = dict(size = x_axis['sizes'][n] if 'sizes' in x_axis else 10),
                            line = dict(width = x_axis['sizes'][n] if 'sizes' in x_axis else 2)
                        ),
                        secondary_y = secondary_y
                              )
            
        if 'scale' in x_axis.keys():
            if x_axis["scale"] == "loglog":
                fig.update_layout(
                    xaxis_title = x_axis_title,
                    yaxis_title = y_axis_title,
                    xaxis=dict(type="log", exponentformat="e"),  
                    yaxis=dict(type="log", exponentformat="e")  
                )
            elif x_axis["scale"] == "logx":
                fig.update_layout(
                    xaxis_title = x_axis_title,
                    yaxis_title = y_axis_title,
                    xaxis=dict(type="log", exponentformat="e"),  
                )
            elif x_axis["scale"] == "logy":
                fig.update_layout(
                    xaxis_title = x_axis_title,
                    yaxis_title = y_axis_title,
                    yaxis=dict(type="log", exponentformat="e")  
                )                
        else:
            fig.update_layout(
                    xaxis_title = x_axis_title,
                    yaxis_title = y_axis_title
                    )
            
        if "xlim" in x_axis.keys():
             fig.update_xaxes(range = x_axis["xlim"])
        if "ylim" in x_axis.keys():
             fig.update_yaxes(range = x_axis["ylim"])
        
    def polar_plot_to_html(self, plot_dictionary, auto_open = False):
        """
        Plot scatter points using Plotly in polar coordinates.

        Parameters
        ----------
        plot_dictionary : dict
            A dictionary containing data for plotting in polar coordinates. The dictionary should
            contain keys 'r_axis_title' and 'theta_axis_title' with their corresponding values and optional
            markers, lines, sizes, and legends.
        auto_open : bool, optional
            If True, automatically open the plot in a browser (default is False).

        Example
        -------
        
        .. code-block:: python
            
            plot_dictionary = {
                'Coordinate Radius': {
                    'values': [r_values_1, r_values_2],
                    'markers': ['markers', 'lines'],
                    'sizes': [1, 5],
                    'legend': ['Data 1', 'Data 2']
                },
                'Coordinate Theta': {
                    'values': [theta_values_1, theta_values_2],
                }
            }

        Example Usage
        -------------
        >>> plot.polar_plot_to_html(plot_dictionary)
        """
        try:
            import plotly.graph_objects as go
            from plotly.offline import plot
        except ImportError as e:
            print(e)
            return

        self.print_text("info", "\nPlotting polar plot...")
        fig = go.Figure()

        # Get axis titles
        r_axis_title = list(plot_dictionary.keys())[0]
        theta_axis_title = list(plot_dictionary.keys())[1]

        r_axis = plot_dictionary[r_axis_title]
        theta_axis = plot_dictionary[theta_axis_title]

        # Plotting
        self.__polar_plot(go, fig, r_axis, theta_axis)
        
        export_path = self.export_path_check()
        output_path = os.path.join(export_path, self.files_name_tag)
        self.print_text("check", f"\n\tHTML Plot produced in : {output_path}")
        plot(fig, filename = output_path, auto_open = auto_open)

    def __polar_plot(self, go, fig, r_axis, theta_axis):
        for n, (r, theta) in enumerate(zip(r_axis['values'], theta_axis['values'])):
            mode = r_axis.get('markers', ['markers'])[n]
            if mode == 'lines':
                fig.add_trace(go.Scatterpolar(
                    r=r,
                    theta=theta,
                    mode='lines',
                    name=r_axis['legend'][n] if 'legend' in r_axis else None,
                    line=dict(width=r_axis['sizes'][n] if 'sizes' in r_axis else 2)
                ))
            else:  # default
                fig.add_trace(go.Scatterpolar(
                    r=r,
                    theta=theta,
                    mode='markers',
                    name=r_axis['legend'][n] if 'legend' in r_axis else None,
                    marker=dict(size=r_axis['sizes'][n] if 'sizes' in r_axis else 10)
                ))
        