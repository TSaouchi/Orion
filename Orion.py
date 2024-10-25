### Path and Char manipulation
import os
import time

### Math
import numpy as np
import copy
import scipy as spy

import Core as Orion
from DataProcessor import Processor
from Utils import *

if __name__ == "__main__":
    # ========================= Cases configuration ============================
    # ============================== Read inputs ===============================
    base = Orion.Base()
    
    nzone = 5
    ninstant = 5
    nvar = 10
    n = 1e2
    
    zones = [f"Zone_{i}" for i in range(nzone)]
    instants = [f"Instant_{i}" for i in range(ninstant)]
    var1 = [f"var_{i}" for i in range(0, nvar)]
    var1_value = nvar*[da.random.random(n, chunks = 'auto')] 
    base.init(zones, instants, var1, var1_value)
    
    plotter = Orion.Plotter(base)
    plotter.run()
