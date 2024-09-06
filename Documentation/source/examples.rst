.. include:: references.rst

Examples
********
This section presents simple and common examples of code usage. For more details and method flexibility, feel free to browse the method-specific documentation.

Read and Write
==============

Readers
-------


Read HDF5 dataset files
^^^^^^^^^^^^^^^^^^^^^^^

Read Matlab dataset files
^^^^^^^^^^^^^^^^^^^^^^^^^

Read Excel files
^^^^^^^^^^^^^^^^

Read tabulated .dat, .csv... files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Writers
-------

The default output directory name is **output**, and the default output file names are **output_result.extension** or **output_result_inst<instant number>.extension** for result files. To change these defaults, use the Writer arguments ``dir_name_tag`` and ``files_name_tag`` for the output directory name and output file names, respectively.

Computing
=========

The following example shows how to calculate physical quantities and signal processing.

Physics
-------

Signal processing
-----------------


Interpolate data
================

Plotting
========

The ``Plot`` class gather methods for plotting data, one can use the bellow code snippets to generate dynamic |PlotltyGithub| graphs as depicted in the graphs below. 

Cartesian plot
--------------

.. raw:: html

    <div style="display: flex; justify-content: center; flex-direction: column; align-items: center;">
        <div>
            <button class="button" onclick="loadHtml('_static\\pictures\\rotate_blade.html', 'plotContainer3')">Load Plot</button>
        </div>
        <div id="plotContainer3" style="display: none;"></div>
    </div>

.. code-block:: python

    # Optional, merging bases to avoid iterating over zones
    base_rotate = Processor(base_rotate).merge()
    base_original = Processor(base_original).merge()

    # Plotting dictionary 
    plot_dictionary = {
        'Coordinate X' : {
            'values' : [base_original[0][0]['CoordinateX'].ravel(), base_rotate[0][0]['CoordinateX'].ravel()],
            'markers' : 2*['markers'],
            'legend' : ['original', 'rotate']
        },
        'Coordinate Y' : {
            'values' : [base_original[0][0]['CoordinateY'].ravel(), base_rotate[0][0]['CoordinateY'].ravel()],
        },
        'Coordinate Z' : {
            'values' : [base_original[0][0]['CoordinateZ'].ravel(), base_rotate[0][0]['CoordinateZ'].ravel()],
        }
    }
    #Plot
    Plot(input_common_path).cartesian_plot_to_html(plot_dictionary)

Polar plot
----------

.. raw:: html

    <div style="display: flex; justify-content: center; flex-direction: column; align-items: center;">
        <div>
            <button class="button" onclick="loadHtml('_static\\pictures\\polar_plot.html', 'plotContainer')">Load Plot</button>
        </div>
        <div id="plotContainer" style="display: none;"></div>
    </div>

.. code-block:: python 

    # Plotting dictionary
    plot_dictionary = {
        'Coordinate R' : {
            'values' : [np.arange(0, 90, 1), 
                        np.arange(0, 180, 1), 
                        np.arange(0, 45, 1)],
            'markers' : ['markers'] + 2*['lines'],
            'sizes' : [3, 1, 2],
            'legend' : ['marker', 'line 1', 'line 2']
        },
        'Coordinate Theta' : {
            'values' : [np.arange(0, 90, 1), 
                        np.arange(0, 180, 1), 
                        np.arange(0, 45, 1)],
        }
    }
    #Plot
     Plot(input_common_path).polar_plot_to_html(plot_dictionary)

.. note:: 

    The ``Plot()`` class accepts the following additional arguments:

        - ``dir_name_tag``: The name of the output directory, default **output**
        - ``files_name_tag`` : The name of the html file, default **output_result**.
    
    The methods of the class accepts the following additional argument:

        - ``auto_open`` : Boolean argument if set to ``True`` the graph will open automatically, default set to ``False``.

Practical examples
==================
