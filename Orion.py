# Path and Char manipulation
import os
import time

# Math
import numpy as np
import copy
import scipy as spy

import Core as Orion
from DataProcessor import Processor
from Utils import *
from Debug import PerformanceStats

Reader = Orion.Reader
def tmp_reader_BHG(base_blueprint, Reader):
    base = Orion.Base()
    for zone, config in base_blueprint.items():
        base.add_zone(zone)
        files_dir = list(Reader(config["path"], patterns=config["dir_name_pattern"]
                            )._Reader__generate_file_list(config["path"], 
                                                          patterns=config["dir_name_pattern"]
                                                          ))
        for dir in files_dir:
            try:
                files = list(Reader(os.path.join(config["path"], dir), 
                                    patterns=config["file_name_pattern"]
                                    )._Reader__generate_file_list(
                                        os.path.join(config["path"], dir), 
                                        patterns=config["file_name_pattern"]))
            except: continue
            
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
    return Orion.SharedMethods().variable_mapping_cgns(base)

if __name__ == "__main__":
    # ========================= Cases configuration ============================
    # ============================== Read inputs ===============================
    import dask.array as da
    base = Orion.Base()
    
    nzone = 500
    ninstant = 500
    nvar = 100
    n = 1e8
    zones = [f"Zone_{i}" for i in range(nzone)]
    instants = [f"Instant_{i}" for i in range(ninstant)]
    var1 = [f"var_{i}" for i in range(0, nvar)]
    var1_value = nvar*[da.random.random((n, 1), chunks = 'auto')]    
    
    with PerformanceStats() as stats:
        base.init(zones, instants, var1, var1_value)

    with PerformanceStats() as stats:
        base.compute('var2 =  2')

