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
            "BWI_soft_mode",
            "BWI_hard_mode"
            ],
        "Paths" : [
            r"C:\__sandBox__\Data\13124-846_G60 BMW DVP - Pink noise signal\20 BMW Signale\stimuli_pink_noise_0p1_30Hz_20s\info",
            r"C:\__sandBox__\Data\13124-846_G60 BMW DVP - Pink noise signal\Single_measures_without_topmount\24-0052_RR_G60_1963_R2_Pink_Noise_0p4A_Sinus",
            r"C:\__sandBox__\Data\13124-846_G60 BMW DVP - Pink noise signal\Single_measures_without_topmount\24-0052_RR_G60_1963_R2_Pink_Noise_1p6A_Sinus"
        ],
        "file_name_patterns" : [
            "damper_noise_red_HA_amp<instant>",
            "Pink_Noise_HA_amp<instant>",
            "Pink_Noise_HA_amp<instant>"
            ],
        "Variables": [
            ['t', 'z', 'trigger'],
            ["Temps", "Axial Displacement", "Axial Force"],
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
            
            base_tmp.compute("VelocityZ = np.diff(CoordinateZ)/(TimeValue[1] - TimeValue[0])")
            base.append(base_tmp)
        else:
            base_tmp = Reader(cases["Paths"][nzone], 
                              cases["file_name_patterns"][nzone]).read_ascii(
                        variables = cases["Variables"][nzone],
                        zone_name = [zone])
                
            base_tmp.compute("VelocityZ = np.diff(CoordinateZ)/(TimeValue[1] - TimeValue[0])")
            base_tmp = Processor(base_tmp).reduce(factor=4)
            base.append(base_tmp)

    base = Processor(base).fusion()
    speed = ["amp001p2_spd0100","amp002p4_spd0200","amp006p0_spd0500","amp012p1_spd1000","amp018p1_spd1500"]
    for zone in base.keys():
        base[zone].rename_instant(list(base[zone].keys()), speed)
    # # ============================= Data manipulation ==========================
    #NOTE - Signal BMW vs BWI
    Axis_x = []
    Axis_y= []
    legend = []
    
    for zone, instant in base.items():
        Axis_x.append(base[zone][instant]['TimeValue'].compute() - base[zone][instant]['TimeValue'].compute().min())
        if zone == 'BMW':
            Axis_y.append(1e3*base[zone][instant]['CoordinateZ'].compute())
        else:
            Axis_y.append(base[zone][instant]['CoordinateZ'].compute())
        legend.append(f"{zone}_{instant}")
    
    n = len(legend)
        
    plot_dictionary = {
        'Time (s) - Shifted to start at zero' : {
            'values' : Axis_x,
            'markers' : n*['lines'],
            'legend' : legend,
            'sizes' : n*[2]
        },
        'Displacement/BMW maximum dispalcement' : {
            'values' : Axis_y
        },
    }
    output_path = r"C:\__sandBox__\Data\13124-846_G60 BMW DVP - Pink noise signal"
    Ploter = Orion.Plot
    Ploter(output_path, dir_name_tag = r"Reporting\Input", files_name_tag = f"Time_domain_Input_BMWvsBWI_PID-Sinus").cartesian_plot_to_html(plot_dictionary)
    
    # FFT BMW vs BWI
    # Signal BMW vs BWI
    base = Processor(base).fft(decomposition_type='complex', frequencies_band = (None, 35))
    Axis_x = []
    Axis_y= []
    legend = []
    
    for zone, instant in base.items():
        Axis_x.append(base[zone][instant]['Frequency'].compute())
        if zone == 'BMW':
            Axis_y.append(1e3*np.abs(base[zone][instant]['CoordinateZ_complex'].compute()))
        else:
            Axis_y.append(np.abs(base[zone][instant]['CoordinateZ_complex'].compute()))
        legend.append(f"{zone}_{instant}")
    
    n = len(legend)
        
    plot_dictionary = {
        'Frequency (Hz)' : {
            'values' : Axis_x,
            'markers' : n*['lines'],
            'legend' : legend,
            'sizes' : n*[2]
        },
        'FFt(Displacement) (mm)' : {
            'values' : Axis_y
        },
    }
    output_path = r"C:\__sandBox__\Data\13124-846_G60 BMW DVP - Pink noise signal"
    Ploter = Orion.Plot
    Ploter(output_path, dir_name_tag = r"Reporting\Input", files_name_tag = f"Frequency_domain_Input_Magnitude_BMWvsBWI_PID-Sinus").cartesian_plot_to_html(plot_dictionary)
    
    Axis_x = []
    Axis_y= []
    legend = []
    
    for zone, instant in base.items():
        Axis_x.append(base[zone][instant]['Frequency'].compute())
        if zone == 'BMW':
            Axis_y.append(np.degrees(np.angle(base[zone][instant]['CoordinateZ_complex'].compute())))
        else:
            Axis_y.append(np.degrees(np.angle(base[zone][instant]['CoordinateZ_complex'].compute())))
        legend.append(f"{zone}_{instant}")
    
    n = len(legend)
        
    plot_dictionary = {
        'Frequency (Hz)' : {
            'values' : Axis_x,
            'markers' : n*['lines'],
            'legend' : legend,
            'sizes' : n*[2]
        },
        'FFt(Displacement) (mm)' : {
            'values' : Axis_y
        },
    }
    output_path = r"C:\__sandBox__\Data\13124-846_G60 BMW DVP - Pink noise signal"
    Ploter = Orion.Plot
    Ploter(output_path, dir_name_tag = r"Reporting\Input", files_name_tag = f"Frequency_domain_Input_Phase_BMWvsBWI_PID-Sinus").cartesian_plot_to_html(plot_dictionary)
    
    # #FIXME -  Smoothing
    # base = Processor(base).smooth(window=[2, 100, 2], order=1)
    #NOTE - Force over Displacement
    Axis_x = []
    Axis_y= []
    legend = []
    
    for zone, instant in base.items():
        if zone != 'BMW':
            Axis_x.append(base[zone][instant]['Frequency'].compute())
            Axis_y.append(np.abs(base[zone][instant]['ForceZ_complex'].compute()/base[zone][instant]['CoordinateZ_complex'].compute()))
            legend.append(f"{zone}_{instant}")
    
    n = len(legend)
        
    plot_dictionary = {
        'Frequency (Hz)' : {
            'values' : Axis_x,
            'markers' : n*['lines'],
            'legend' : legend,
            'sizes' : n*[2]
        },
        'Frequency Response Function |H(f)| (N/mm)' : {
            'values' : Axis_y
        },
    }
    output_path = r"C:\__sandBox__\Data\13124-846_G60 BMW DVP - Pink noise signal"
    Ploter = Orion.Plot
    Ploter(output_path, dir_name_tag = r"Reporting\Output", files_name_tag = f"Frequency_domain_Output_Magnitude-F-S_BWI_PID-Sinus").cartesian_plot_to_html(plot_dictionary)
    
    Axis_x = []
    Axis_y= []
    legend = []
    
    for zone, instant in base.items():
        if zone != 'BMW':
            Axis_x.append(base[zone][instant]['Frequency'].compute())
            Axis_y.append(np.degrees(np.angle(base[zone][instant]['ForceZ_complex'].compute()/base[zone][instant]['CoordinateZ_complex'].compute())))
            legend.append(f"{zone}_{instant}")
    
    n = len(legend)
        
    plot_dictionary = {
        'Frequency (Hz)' : {
            'values' : Axis_x,
            'markers' : n*['markers'],
            'legend' : legend,
            'sizes' : n*[2]
        },
        'Frequency Response Function arg(H(f)) (°)' : {
            'values' : Axis_y
        },
    }
    output_path = r"C:\__sandBox__\Data\13124-846_G60 BMW DVP - Pink noise signal"
    Ploter = Orion.Plot
    Ploter(output_path, dir_name_tag = r"Reporting\Output", files_name_tag = f"Frequency_domain_Output_Phase-F-S_BWI_PID-Sinus").polar_plot_to_html(plot_dictionary)
    
    #NOTE - Force over Velocity
    Axis_x = []
    Axis_y= []
    legend = []
    
    for zone, instant in base.items():
        if zone != 'BMW':
            Axis_x.append(base[zone][instant]['Frequency'].compute())
            Axis_y.append(np.abs(base[zone][instant]['ForceZ_complex'].compute()/base[zone][instant]['VelocityZ_complex'].compute()))
            legend.append(f"{zone}_{instant}")
    
    n = len(legend)
        
    plot_dictionary = {
        'Frequency (Hz)' : {
            'values' : Axis_x,
            'markers' : n*['lines'],
            'legend' : legend,
            'sizes' : n*[2]
        },
        'Impedande |Z(f)| (N.s/mm)' : {
            'values' : Axis_y
        },
    }
    output_path = r"C:\__sandBox__\Data\13124-846_G60 BMW DVP - Pink noise signal"
    Ploter = Orion.Plot
    Ploter(output_path, dir_name_tag = r"Reporting\Output", files_name_tag = f"Frequency_domain_Output_Magnitude-F-V_BWI_PID-Sinus").cartesian_plot_to_html(plot_dictionary)
    
    Axis_x = []
    Axis_y= []
    legend = []
    
    for zone, instant in base.items():
        if zone != 'BMW':
            Axis_x.append(base[zone][instant]['Frequency'].compute())
            Axis_y.append(np.degrees(np.angle(base[zone][instant]['ForceZ_complex'].compute() - base[zone][instant]['VelocityZ_complex'].compute())))
            legend.append(f"{zone}_{instant}")
    
    n = len(legend)
        
    plot_dictionary = {
        'Frequency (Hz)' : {
            'values' : Axis_x,
            'markers' : n*['markers'],
            'legend' : legend,
            'sizes' : n*[2]
        },
        'Impedande arg(Z(f)) (°)' : {
            'values' : Axis_y
        },
    }
    output_path = r"C:\__sandBox__\Data\13124-846_G60 BMW DVP - Pink noise signal"
    Ploter = Orion.Plot
    Ploter(output_path, dir_name_tag = r"Reporting\Output", files_name_tag = f"Frequency_domain_Output_Phase-F-V_BWI_PID-Sinus").polar_plot_to_html(plot_dictionary)