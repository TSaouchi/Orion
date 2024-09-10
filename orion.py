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
            base_tmp.compute("VelocityZ = np.diff(CoordinateZ)/(TimeValue[1] - TimeValue[0])")
            base.append(base_tmp)
        else:
            base_tmp = Reader(cases["Paths"][nzone], 
                              cases["file_name_patterns"][nzone]).read_ascii(
                        variables = cases["Variables"][nzone],
                        zone_name = [zone])
                
            base_tmp.compute("VelocityZ = np.diff(CoordinateZ)/(TimeValue[1] - TimeValue[0])")
            #* INFO - Resampling of the measure
            base_tmp = Processor(base_tmp).reduce(factor = 4)
            base.append(base_tmp)

    base = Processor(base).fusion()
    speed = ["amp001p2_spd0100"]
    for zone in base.keys():
        base[zone].rename_instant(list(base[zone].keys()), speed)
    ## ============================= Data manipulation ==========================
    #! Input signal BMW vs BWI 
    #* Time domain  
    Axis_x = []
    Axis_y= []
    legend = []
    
    for zone, instant in base.items():
        Axis_x.append(base[zone][instant]['TimeValue'].compute() - base[zone][instant]['TimeValue'].compute().min())
        if zone == 'BMW':
            Axis_y.append(base[zone][instant]['CoordinateZ'].compute())
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
    
    #* PSD
    psd_base = Processor(base).psd(frequencies_band = (0, 30))
    Axis_x = []
    Axis_y= []
    legend = []
    
    for zone, instant in psd_base.items():
        Axis_x.append(psd_base[zone][instant]['Frequency'].compute())
        if zone == 'BMW':
            Axis_y.append(psd_base[zone][instant]['CoordinateZ'].compute())
        else:
            Axis_y.append(psd_base[zone][instant]['CoordinateZ'].compute())
        legend.append(f"{zone}_{instant}")
    
    n = len(legend)
    
    plot_dictionary = {
        'Frequency (Hz)' : {
            'values' : Axis_x,
            'markers' : n*['lines'],
            'legend' : legend,
            'sizes' : n*[2]
        },
        'PSD (mm^2/Hz)' : {
            'values' : Axis_y
        },
    }
    output_path = r"C:\__sandBox__\Data\13124-846_G60 BMW DVP - Pink noise signal"
    Ploter = Orion.Plot
    Ploter(output_path, dir_name_tag = r"Reporting\Input", files_name_tag = f"Frequency_domain_Input_PSD_BMWvsBWI_PID-Sinus").cartesian_plot_to_html(plot_dictionary)
    
    #* FFT 
    base = Processor(base).fft(decomposition_type='mod/phi', frequencies_band = (0.1, 30))
    Axis_x = []
    Axis_y= []
    legend = []
    
    for zone, instant in base.items():
        Axis_x.append(base[zone][instant]['Frequency'].compute())
        if zone == 'BMW':
            Axis_y.append(base[zone][instant]['CoordinateZ_mag'].compute())
        else:
            Axis_y.append(base[zone][instant]['CoordinateZ_mag'].compute())
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
            Axis_y.append(np.degrees(base[zone][instant]['CoordinateZ_phase'].compute()))
        else:
            Axis_y.append(np.degrees(base[zone][instant]['CoordinateZ_phase'].compute()))
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
    
    #! FFT measure BWI  
    #* FFT : Force over Displacement
    Axis_x = []
    Axis_y= []
    legend = []
    
    for zone, instant in base.items():
        if zone != 'BMW':
            Axis_x.append(base[zone][instant]['Frequency'].compute())
            Axis_y.append(20*np.log10(np.abs(base[zone][instant]['ForceZ_mag'].compute()/(1e-3*base[zone][instant]['CoordinateZ_mag'].compute()))))
            legend.append(f"{zone}_{instant}")
    
    #FIXME -  Smoothing
    base_smooth = Processor(base).smooth(window=[31, 150, 5], order=2)
    Axis_x.append(base_smooth[1][0]['Frequency'].compute())
    Axis_y.append(20*np.log10(np.abs(base_smooth[1][0]['ForceZ_mag'].compute()/(1e-3*base_smooth[1][0]['CoordinateZ_mag'].compute()))))
    legend.append(f"{legend[0]}_smoothed")
    
    n = len(legend)
    plot_dictionary = {
        'Frequency (Hz)' : {
            'values' : Axis_x,
            'markers' : n*['lines'],
            'legend' : legend,
            'sizes' : n*[2],
            'scale' : 'logx'
        },
        'FRF 20log10(|H(f)|) (dB)' : {
            'values' : Axis_y
        },
    }
    output_path = r"C:\__sandBox__\Data\13124-846_G60 BMW DVP - Pink noise signal"
    Ploter = Orion.Plot
    Ploter(output_path, dir_name_tag = r"Reporting\Output", files_name_tag = f"Frequency_domain_Output_Magnitude-F-S_BWI_PID-Sinus").cartesian_plot_to_html(plot_dictionary)
    
    
    Axis_x = []
    Axis_y= []
    legend = []
    mod = lambda x : np.degrees(np.mod(x + np.pi, 2 * np.pi) - np.pi)
    for zone, instant in base.items():
        if zone != 'BMW':
            Axis_x.append(base[zone][instant]['Frequency'].compute())
            phase_diff = base[zone][instant]['ForceZ_phase'].compute() - base[zone][instant]['CoordinateZ_phase'].compute()
            base[zone][instant].add_variable("phase_mod", mod(phase_diff))
            Axis_y.append(mod(phase_diff))
            legend.append(f"{zone}_{instant}")
    
    #FIXME -  Smoothing
    base_smooth = Processor(base).smooth(window=[4, 500, 5], order=1)
    Axis_x.append(base_smooth[1][0]['Frequency'].compute())
    Axis_y.append(base_smooth[1][0]['phase_mod'].compute())
    legend.append(f"{legend[0]}_smoothed")
    n = len(legend)
    plot_dictionary = {
        'Frequency (Hz)' : {
            'values' : Axis_x,
            'markers' : n*['lines'],
            'legend' : legend,
            'sizes' : n*[2]
        },
        'FRF arg(H(f)) (°)' : {
            'values' : Axis_y
        },
    }
    output_path = r"C:\__sandBox__\Data\13124-846_G60 BMW DVP - Pink noise signal"
    Ploter = Orion.Plot
    Ploter(output_path, dir_name_tag = r"Reporting\Output", files_name_tag = f"Frequency_domain_Output_Phase-F-S_BWI_PID-Sinus").cartesian_plot_to_html(plot_dictionary)