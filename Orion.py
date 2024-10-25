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
from Debug import PerformanceStats


if __name__ == "__main__":
    # ========================= Cases configuration ============================
    # ============================== Read inputs ===============================
    def generate_sample_data():
        base = Orion.Base()
        nzone = 5
        ninstant = 5
        n = 10
        zones = [f"Zone_{i}" for i in range(nzone)]
        instants = [f"Instant_{i}" for i in range(ninstant)]
        base.init(zones, instants)
        
        var1 = [f"var_{i}" for i in range(0, 5)]
        var2 = [f"var_{i}" for i in range(50, 60)]
        var3 = [f"var_{i}" for i in range(60, 70)]
        
        for zone in zones[0:2]:
            for instant in instants:
                for var in var1:
                    base[zone][instant].add_variable(var, 
                                                     np.sin(da.random.random(n, 1))
                                                     )
        for zone in zones[2:3]:
            for instant in instants:
                for var in var2:
                    base[zone][instant].add_variable(var, 
                                                     np.cos(da.random.random(n, 1))
                                                     )
        for zone in zones[3:5]:
            for instant in instants:
                for var in var3:
                    base[zone][instant].add_variable(var, 
                                                     np.sin(da.random.random(n, 1))*\
                                                         np.cos(da.random.random(n, 1))
                                                     )
        return base
    base = generate_sample_data()
    
    plotter = Orion.Plotter(base)
    plotter.run()
