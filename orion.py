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
            r"C:\__sandBox__\Data\pink noise\20 BMW Signale\stimuli_pink_noise_0p1_30Hz_20s\info",
            r"C:\__sandBox__\Data\pink noise\EWR_13124-846_Front_24-0021_Pink_Noise_0p4A",
            r"C:\__sandBox__\Data\pink noise\EWR_13124-846_Front_24-0021_Pink_Noise_1p6A"
        ],
        "file_name_patterns" : [
            "damper_noise_red_HA_amp<instant>",
            "Pink_Noise_HA_amp<instant>",
            "Pink_Noise_HA_amp<instant>"
            ],
        "Variables": [
            ['t', 'z', 'trigger'],
            ["Temps", "Axial Displacement"],
            ["Temps", "Axial Displacement"]
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
            
            base_tmp.compute("VelocityZ = np.diff(CoordinateZ)/(TimeValues[1] - TimeValues[0])")
            base.append(base_tmp)
            del base_tmp
        else:
            base_tmp = Reader(cases["Paths"][nzone], 
                              cases["file_name_patterns"][nzone]).read_ascii(
                        variables = cases["Variables"][nzone],
                        zone_name = [zone])
                
            base_tmp.compute("VelocityZ = np.diff(CoordinateZ)/(TimeValues[1] - TimeValues[0])")
            base.append(base_tmp)
            del base_tmp
        
    # ============================= Data manipulation ==========================
    base = Processor(base).fusion()
    

    # Function to write the data in column format to a .dat file
    def write_dat_file_in_columns(base, file_path):
        for instant, file_paths in zip(base[0].keys(), file_path):
            variables = base[0][instant].keys()
            
            # Ensure all arrays have the same length, pad with NaNs if necessary
            max_length = max(len(base[0][instant][v].data) for v in variables)
            column_data = []

            for var in variables:
                var = base[0][instant][var].data.compute()
                if len(var) < max_length:
                    # Padding shorter arrays with NaNs
                    padded_var = np.pad(var, (0, max_length - len(var)), constant_values=np.nan)
                    column_data.append(padded_var)
                else:
                    column_data.append(var)
            
            # Stack the arrays column-wise
            stacked_data = np.column_stack(column_data)
            
            # Write to .dat file
            with open(file_paths, 'w') as file:
                # Write the header with variable names
                header = "\t".join(variables)
                file.write(header + "\n")
                
                # Write the data row by row
                np.savetxt(file, stacked_data, delimiter="\t")

    # Call the function
    write_dat_file_in_columns(base, output_file)