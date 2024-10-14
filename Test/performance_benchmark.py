# Math
import sys
import os

# Get the directory of the current script
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import numpy as np
import dask.array as da

import Core as Orion
from Debug import PerformanceStats

if __name__ == "__main__":
    base = Orion.Base()
    
    nzone = 500
    ninstant = 500
    nvar = 100
    n = 1e8
    
    zones = [f"Zone_{i}" for i in range(nzone)]
    instants = [f"Instant_{i}" for i in range(ninstant)]
    var1 = [f"var_{i}" for i in range(0, nvar)]
    var1_value = nvar*[da.random.random((n, 1), chunks = 'auto')]    
    
    print("Init base")
    with PerformanceStats() as stats:
        base.init(zones, instants, var1, var1_value)
    
    value = stats.get_stats()['execution_time']
    expected_value = 30
    try:
        np.testing.assert_(value < expected_value, 
                           msg=f"\t\nValue {value} is not less than {expected_value}")
        print("\t\nValue is less than 40")
    except AssertionError as e:
        print(e)
    
    print("Compute variable in base")
    with PerformanceStats() as stats:
        base.compute('VarMultiplication = var_0 * var_0')
        
    value = stats.get_stats()['execution_time']
    expected_value = 30
    try:
        np.testing.assert_(value < expected_value, 
                           msg=f"\t\nValue {value} is not less than {expected_value}")
        print("\t\nValue is less than 40")
    except AssertionError as e:
        print(e)
