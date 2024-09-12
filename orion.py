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
            'BMW',
            "BWI_soft_mode"
            ],
        "Paths" : [
            r"C:\__sandBox__\Data\13124-846_G60 BMW DVP - Pink noise signal\20 BMW Signale\stimuli_pink_noise_0p1_30Hz_20s\info",
            r"C:\__sandBox__\Data\13124-846_G60 BMW DVP - Pink noise signal\Single_measures_without_topmount\24-0052_RR_G60_1963_R2_Pink_Noise_0p4A_Sinus"
        ],
        "file_name_patterns" : [
            "damper_noise_red_HA_amp001p2_spd0100_info.mat",
            "Pink_Noise_HA_amp001p2_spd0100.dat"
            ],
        "Variables": [
            ['t', 'z', 'trigger'],
            ["Temps", "Axial Displacement", "Axial Force"]
        ]
    }

    # ============================== Read inputs ===============================
    Reader = Orion.Reader
    base = []
    
    for nzone, zone in enumerate(cases["Zones"]):
        if nzone == 0:
            base_tmp = Reader(cases["Paths"][nzone], 
                              cases["file_name_patterns"][nzone]).read_mat(
                                  variables = cases["Variables"][nzone],
                                  zone_name = [zone])
            for zone, instant in base_tmp.items():
                base_tmp[zone][instant].add_variable("CoordinateZ", 1e3*base_tmp[zone][instant]['CoordinateZ'].data)
            # base_tmp.compute("VelocityZ = np.diff(CoordinateZ)/(TimeValue[1] - TimeValue[0])")
            base.append(base_tmp)
        else:
            base_tmp = Reader(cases["Paths"][nzone], 
                              cases["file_name_patterns"][nzone]).read_ascii(
                        variables = cases["Variables"][nzone],
                        zone_name = [zone])
                
            # base_tmp.compute("VelocityZ = np.diff(CoordinateZ)/(TimeValue[1] - TimeValue[0])")
            #* INFO - Resampling of the measure
            base_tmp = Processor(base_tmp).reduce(factor = 4)
            base.append(base_tmp)

    base = Processor(base).fusion()
    speed = ["amp001p2_spd0100"]
    for zone in base.keys():
        base[zone].rename_instant(list(base[zone].keys()), speed)
    ## ============================= Data manipulation ==========================
    # #! Transfer Function 
    # from scipy import signal
    # output = signal.detrend(base[1][0]['ForceZ'].compute())
    # input = signal.detrend(1e-3*base[1][0]['CoordinateZ'].compute())
    # time =  base[1][0]['TimeValue'].compute()
    # _, dict_transfer = Processor.compute_transfer_function(input, output, time, order = (1, 1))
    
    #! Detrending
    detrend_base = Processor(base).fft(decomposition_type='mod/phi', frequencies_band = (0.1, 30))
    detrend_base.compute('H = ForceZ_mag/CoordinateZ_mag')
    detrend_base = Processor(detrend_base).detrend(type='linear')
    Axis_x = []
    Axis_y= []
    legend = []
    
    for zone, instant in base.items():
        if zone != 'BMW':
            Axis_x.append(detrend_base[zone][instant]['Frequency'].compute())
            ratio = detrend_base[zone][instant]['H'].compute()
            Axis_y.append(ratio)
            legend.append(f"{zone}_{instant}_det")
            
    #! FFT measure BWI  
    base = Processor(base).fft(decomposition_type='mod/phi', frequencies_band = (0.1, 30))
    base.compute('H = ForceZ_mag/CoordinateZ_mag')
    # Axis_x = []
    # Axis_y= []
    # legend = []
    
    for zone, instant in base.items():
        if zone != 'BMW':
            Axis_x.append(base[zone][instant]['Frequency'].compute())
            ratio = base[zone][instant]['ForceZ_mag'].compute()/(base[zone][instant]['CoordinateZ_mag'].compute())
            Axis_y.append(ratio)
            legend.append(f"{zone}_{instant}")
    
    #! WARNING  Smoothing
    base_smooth = Processor(base).smooth(window=[31, 150, 5], order=2)
    Axis_x.append(base_smooth[1][0]['Frequency'].compute())
    ratio = base_smooth[1][0]['ForceZ_mag'].compute()/(base_smooth[1][0]['CoordinateZ_mag'].compute())
    Axis_y.append(ratio)
    legend.append(f"{legend[0]}_smoothed")
    
    #! WARNING  linear regression
    base_linear = Processor(base).linear_regression(independent_variable_name = 'Frequency')
    Axis_x.append(base_linear[1][0]['Frequency'].compute())
    ratio = base_linear[1][0]['H'].compute()
    Axis_y.append(ratio)
    legend.append(f"{legend[0]}_linear")
    
    Axis_x.append(base_linear[1][0]['Frequency'].compute())
    ratio = base_linear[1][0]['H'].get_attribute('residuals').compute()
    Axis_y.append(ratio)
    legend.append(f"{legend[0]}_residual")
      
    # Axis_x.append(dict_transfer['frequencies'])
    # ratio = dict_transfer['magnitude']
    # Axis_y.append(ratio - ratio[0])
    # legend.append(f"{legend[0]}_TF")
    
    n = len(legend)
    plot_dictionary = {
        'Frequency (Hz)' : {
            'values' : Axis_x,
            'markers' : n*['lines'],
            'legend' : legend,
            'sizes' : n*[2],
            'scale' : 'logx'
        },
        '|H(f)| (N/mm)' : {
            'values' : Axis_y
        },
    }
    output_path = r"C:\__sandBox__\Data\13124-846_G60 BMW DVP - Pink noise signal"
    Ploter = Orion.Plot
    Ploter(output_path, dir_name_tag = r"Reporting\Output", files_name_tag = f"Test").cartesian_plot_to_html(plot_dictionary)