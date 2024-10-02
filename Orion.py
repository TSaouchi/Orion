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
            "",
            ],
        "Paths" : [
            "",
        ],
        "file_name_patterns" : [
            "",
            ],
        "Variables": [
            [""],
        ]
    }

    # ============================== Read inputs ===============================
    Reader = Orion.Reader
    base = []
    
    for nzone, zone in enumerate(cases["Zones"]):
        base.append(Reader(
            cases["Paths"][nzone], 
            cases["file_name_patterns"][nzone]).read_ascii(
                variables = cases["Variables"][nzone], 
                zone_name = [zone]))

    base = Processor(base).fusion()
    ## ============================= Data manipulation ==========================

