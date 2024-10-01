### Path and Char manipulation
import os

### Math
import numpy as np
import scipy as spy

import Core as Orion
from DataProcessor import Processor

if __name__ == "__main__":
    # ========================= Cases configuration ============================
    base_blueprint = {
        "EWR12857_136" : {
            "path" : r"C:\__sandBox__\Data\SARC_Gen2_HydraulicTest_Hydac\EWR12857_136",
            "file_name_pattern" : "Valve Hydac*"
        }
    }

    # ============================== Read inputs ===============================
    Reader = Orion.Reader
    base = Orion.Base()
    
    for zone, config in base_blueprint.items():
        base.add_zone(zone)
        files_dir = list(Reader(config["path"], patterns=config["file_name_pattern"]
                            )._Reader__generate_file_list(config["path"], 
                                                          patterns=config["file_name_pattern"]
                                                          ))
        for dir in files_dir:
            files = list(Reader(os.path.join(config["path"], dir)
                                )._Reader__generate_file_list(os.path.join(config["path"], dir)))
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
    base = Orion.SharedMethods().variable_mapping_cgns(base)
    ## ============================= Data manipulation ==========================
    base[0][0].compute("lala = 5")