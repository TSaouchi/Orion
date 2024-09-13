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
    
    expression = "VelocityZ = np.diff(CoordinateZ)/(TimeValue[1] - TimeValue[0])"
    for nzone, zone in enumerate(cases["Zones"]):
        if nzone == 0:
            base_tmp = Reader(cases["Paths"][nzone], 
                              cases["file_name_patterns"][nzone]).read_mat(
                                  variables = cases["Variables"][nzone],
                                  zone_name = [zone])
            for zone, instant in base_tmp.items():
                base_tmp[zone][instant].add_variable("CoordinateZ", 1e3*base_tmp[zone][instant]['CoordinateZ'].data)
            base_tmp.compute(expression)
            base.append(base_tmp)
        else:
            base_tmp = Reader(cases["Paths"][nzone], 
                              cases["file_name_patterns"][nzone]).read_ascii(
                        variables = cases["Variables"][nzone],
                        zone_name = [zone])
                
            base_tmp.compute(expression)
            #* INFO - Resampling of the measure
            base_tmp = Processor(base_tmp).reduce(factor = 4)
            base.append(base_tmp)

    base = Processor(base).fusion()
    ## ============================= Data manipulation ==========================