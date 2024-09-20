### Path and Char manipulation
import os

### Math
import numpy as np
import scipy as spy

import Core as Orion
from DataProcessor import Processor

if __name__ == "__main__":
    # ========================= Cases configuration ============================
    cases = {
        "Zones" : [
            "",
            ],
        "Paths" : [
            "",
        ],
        "file_name_patterns" : [
            "",
            ],
        "Variables": [
            ["Temps", "Axial Displacement", "Axial Force"],
            ["Temps", "Axial Displacement", "Axial Force"]
        ]
    }

    # ============================== Read inputs ===============================
    Reader = Orion.Reader
    base = []
    
    for nzone, zone in enumerate(cases["Zones"]):
        base.append(Reader(
            cases["Paths"][nzone], 
            cases["file_name_patterns"][nzone]).read_ascii(
                variables = cases["Variables"][nzone], 
                zone_name = [zone]))

    base = Processor(base).fusion()
    ## ============================= Data manipulation ==========================
    # Create a simple sine wave with noise
    sampling_rate = 1000  # Hz
    t = np.linspace(0, 1, sampling_rate)
    input_signal1 = np.sin(2 * np.pi * 20 * t)  # 5 Hz sine wave
    input_signal = input_signal1 + np.sin(2 * np.pi * 5 * t)

    base = Orion.Base()
    base.init()
    base[0][0].add_variable('TimeValue', t)
    base[0][0].add_variable('input_signal', input_signal)
    base[0][0].add_variable('input_signal1', input_signal1)
    
    filter_config = {
        'cutoff' : (1, 30),
        'btype' : 'stop',
        'order' : 1,
        'sampling_rate' : sampling_rate
    }
    base_filter = Processor(base).filter(**filter_config)
    
    plot_dictionary = {
        'Time (s) - Shifted to start at zero' : {
            'values' : [base[0][0][0].data, base_filter[0][0][0].data, base_filter[0][0][0].data],
            'markers' : 3*['lines'],
            'legend' : ["original", "filtered", "input"],
            'sizes' : 3*[2]
        },
        'Displacement/BMW maximum dispalcement' : {
            'values' : [base[0][0][1].data, base_filter[0][0][1].data,base[0][0][2].data],
        },
    }
    output_path = r"C:\__sandBox__\Data\13124-846_G60 BMW DVP - Pink noise signal"
    Ploter = Orion.Plot
    Ploter(output_path).cartesian_plot_to_html(plot_dictionary)
