import numpy as np

import Core as Orion
from DataProcessor import Processor

# Initialize the base object
base = Orion.Base()
base.init(['zone_1', 'zone_2'], ['instant_1', 'instant_2'])

# Explore the base
base.show()

# Add variables to the instant
base["zone_1"]["instant_1"].add_variable("Velocity", np.random.randint(1, 101, size=(8, 30)))
base["zone_1"]["instant_1"].add_variable("Pressure", np.random.randint(1, 101, size=(8, 30)))
base["zone_1"]["instant_1"]["Velocity"].set_attribute("Unit", "m/s")

base[0][0].add_variable("Velocity", np.random.randint(1, 101, size=(8, 30)))
base[0][0].add_variable("Pressure", np.random.randint(1, 101, size=(8, 30)))
base[0][0][0].set_attribute("Unit", "m/s")

# Explore the base
base.show()
base[0][0][0]._attributes

# Compute accross all the base using literal expressions
base[0][0].add_variable("Density", 1.08)
base.compute("Ratio = Pressure/(Density*pow(Velocity, 2))")

# Explore the base
base.show(stats = True)

# Compute 
base.compute("Element_number = 5")

# Explore the base
base.show(stats = True) 