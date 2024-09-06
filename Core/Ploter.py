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
    A class for grouping plotting methods
    """
    def __init__(self, path, dir_name_tag = "output", files_name_tag = "output_result"):

        self.dir_name_tag = dir_name_tag
        self.files_name_tag = files_name_tag
        self.path = path

    def cartesian_plot_to_html(self, plot_dictionary, auto_open = False):
        """
        Plot scatter points using Plotly.

        :param plot_dictionary: A dictionary containing data for plotting.
        :type plot_dictionary: dict

        .. note::
            - The dictionary should contain keys ``x_axis_title``, ``y_axis_title``, and optionally ``z_axis_title``.
            - Each key's corresponding value should be a dictionary with keys ``values``, ``markers``, ``lines``, ``sizes``, and ``legend``.
            - ``values`` should contain a list of values for the axis.
            - ``markers`` (optional) should specify the marker style for each set of values and should be provided in the first dictionary.
            It can take the following options:
                - ``markers``: markers only (default)
                - ``markers+lines``: markers and lines
                - ``lines``: lines only
            - ``lines`` (optional) should specify the line style for each set of values and should be provided in the first dictionary.
            It can take the following options:
                - ``lines``: lines only
                - ``markers+lines``: markers and lines
            - ``sizes`` (optional) should specify the marker size for each set of values and should be provided in the first dictionary.
            - ``legend`` (optional) should specify the legend for each set of values and should be provided in the first dictionary.

            - Example of input dictionary:

            .. code-block:: python

                plot_dictionary = {
                    'Coordinate X' : {
                        'values' : [base[0][0]['CoordinateX'].ravel(),
                                    meridiane['HUB']['CoordinateX'],
                                    meridiane['SHROUD']['CoordinateX']],
                        'markers' : ['markers', 'lines', 'lines'],
                        'sizes' : [1, 5, 10],
                        'legend' : ['blade', 'hub', 'shroud']
                    },
                    'Coordinate Y' : {
                        'values' : [base[0][0]['CoordinateY'].ravel(),
                                    meridiane['HUB']['CoordinateY'],
                                    meridiane['SHROUD']['CoordinateY']],
                    },
                    'Coordinate Z' : {
                        'values' : [base[0][0]['CoordinateZ'].ravel(),
                                    meridiane['HUB']['CoordinateZ'],
                                    meridiane['SHROUD']['CoordinateZ']],
                    }
                }

            - Example usage:

            .. code-block:: python

                cartesian_plot_to_html(plot_dictionary)
        """
        try:
            import plotly.graph_objects as go
            from plotly.offline import plot
        except ImportError as e:
            print(e)
            return

        self.print_text("info", "\nPlotting cartesian plot...")
        fig = go.Figure()

        #:Get axis titles
        x_axis_title = list(plot_dictionary.keys())[0]
        y_axis_title = list(plot_dictionary.keys())[1]
        z_axis_title = list(plot_dictionary.keys())[2] if len(plot_dictionary) == 3 else None

        x_axis = plot_dictionary[x_axis_title]
        y_axis = plot_dictionary[y_axis_title]

        #:Plotting for 2D
        if z_axis_title is None:
            for n, (x, y) in enumerate(zip(x_axis['values'], y_axis['values'])):
                mode = x_axis.get('markers', ['markers'])[n]
                if mode == 'lines':
                    fig.add_trace(go.Scatter(
                        x = x,
                        y = y,
                        mode = 'lines',
                        name = x_axis['legend'][n] if 'legend' in x_axis else None,
                        line = dict(width = x_axis['sizes'][n] if 'sizes' in x_axis else 2)
                    ))
                elif mode == 'markers':
                    fig.add_trace(go.Scatter(
                        x = x,
                        y = y,
                        mode = 'markers',
                        name = x_axis['legend'][n] if 'legend' in x_axis else None,
                        marker = dict(size = x_axis['sizes'][n] if 'sizes' in x_axis else 10)
                    ))
                else:
                    fig.add_trace(go.Scatter(
                        x = x,
                        y = y,
                        mode = 'markers+lines',
                        name = x_axis['legend'][n] if 'legend' in x_axis else None,
                        marker = dict(size = x_axis['sizes'][n] if 'sizes' in x_axis else 10),
                        line = dict(width = x_axis['sizes'][n] if 'sizes' in x_axis else 2)
                    ))
            fig.update_layout(xaxis_title = x_axis_title,
                              yaxis_title = y_axis_title)

        #:Plotting for 3D
        else:
            z_axis = plot_dictionary[z_axis_title]
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
            fig.update_layout(
                scene = dict(
                    xaxis_title = x_axis_title,
                    yaxis_title = y_axis_title,
                    zaxis_title = z_axis_title
                    )
                )
        export_path = self.export_path_check()
        output_path = os.path.join(export_path, self.files_name_tag)
        self.print_text("check", f"\n\tHTML Plot produced in : {output_path}")
        plot(fig, filename = output_path, auto_open = auto_open)
        
    def polar_plot_to_html(self, plot_dictionary, auto_open = False):
        """
        Plot scatter points using Plotly in polar coordinates.

        :param plot_dictionary: A dictionary containing data for plotting.
        :type plot_dictionary: dict

        .. note::
            - The dictionary should contain keys ``r_axis_title`` and ``theta_axis_title``.
            - Each key's corresponding value should be a dictionary with keys ``values``, ``phi_values``, ``markers``, ``lines``, ``sizes``, and ``legend``.
            - ``values`` should contain a list of radial values for the axis.
            - ``phi_values`` should contain a list of azimuthal values for the axis.
            - ``markers`` (optional) should specify the marker style for each set of values and should be provided in the first dictionary.
            It can take the following options:
                - ``markers``: markers only (default)
                - ``markers+lines``: markers and lines
                - ``lines``: lines only
            - ``lines`` (optional) should specify the line style for each set of values and should be provided in the first dictionary.
            It can take the following options:
                - ``lines``: lines only
                - ``markers+lines``: markers and lines
            - ``sizes`` (optional) should specify the marker size for each set of values and should be provided in the first dictionary.
            - ``legend`` (optional) should specify the legend for each set of values and should be provided in the first dictionary.

            - Example of input dictionary:

            .. code-block:: python

                plot_dictionary = {
                    'Coordinate Radius' : {
                        'values' : [base[0][0]['CoordinateX'].ravel(),
                                    meridiane['HUB']['CoordinateX'],
                                    meridiane['SHROUD']['CoordinateX']],
                        'markers' : ['markers', 'lines', 'lines'],
                        'sizes' : [1, 5, 10],
                        'legend' : ['blade', 'hub', 'shroud']
                    },
                    'Coordinate Theta' : {
                        'values' : [base[0][0]['CoordinateY'].ravel(),
                                    meridiane['HUB']['CoordinateY'],
                                    meridiane['SHROUD']['CoordinateY']],
                    }
                }

            - Example usage:

            .. code-block:: python

                polar_plot_to_html(plot_dictionary)
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
        export_path = self.export_path_check()
        output_path = os.path.join(export_path, self.files_name_tag)
        self.print_text("check", f"\n\tHTML Plot produced in : {output_path}")
        plot(fig, filename = output_path, auto_open = auto_open)