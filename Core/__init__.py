# Metadata
__title__ = 'Orion'
__description__ = 'A package for data processing, system simulation, and CFD post-processing.'
__version__ = '1.0.0_beta'
__author__ = 'Toufik Saouchi'
__author_email__ = 'toufik.saouchi@gmail.com'
__license__ = '[Your License]'
__github_url__='https://github.com/TSaouchi/Orion'

# Default variables
DEFAULT_NAME = 'Orion'
DEFAULT_SIMULATION_TYPE = '1D'
DEFAULT_ZONE_NAME = ['Zone']
DEFAULT_INSTANT_NAME = ['Instant']
DEFAULT_VERBOSE = False
DEFAULT_SIMULATION_SETTINGS = {
    'tolerance': 1e-5,
    'max_iterations': 1000
}
DEFAULT_TIME_NAME = ['TimeValue']
DEFAULT_COORDINATE_NAMES = ['CoordinateX','CoordinateY','CoordinateZ']

DEFAULT_VAR_SYNONYMS = {

            DEFAULT_TIME_NAME[0] : ['t', 'Temps', 'Time', 'time'],
            DEFAULT_COORDINATE_NAMES[0] : ['X', 'x'],
            DEFAULT_COORDINATE_NAMES[1] : ['Y', 'y'],
            DEFAULT_COORDINATE_NAMES[2] : ['Z', 'z', 'Axial_Displacement', 'Axial Displacement'],

            'VectorNormalX': ['normal_CoordinateX', 'nx'],
            'VectorNormalY': ['normal_CoordinateY', 'ny'],
            'VectorNormalZ': ['normal_CoordinateZ', 'nz'],

            'CoordinateR': ['R', 'r'],
            'CoordinateTheta': ['Theta', 'theta'],
            'CoordinatePhi': ['Phi', 'phi'],

            'VelocityMagnitude' : ['V', 'U'],
            'VelocityX': ['vx', 'Vx', 'u', 'x_velocity'],  #:x-component of momentum
            'VelocityY': ['vy', 'Vy', 'v', 'y_velocity'],
            'VelocityZ': ['vz', 'Vz', 'w', 'z_velocity', 'Axial_Velocity', 'Axial Velocity'],
            'RotatingVelocityMagnitude' : ['W'],
            'RotatingVelocityX': ['Wx', 'wx'],  #:x-component of momentum
            'RotatingVelocityY': ['Wy', 'wy'],
            'RotatingVelocityZ': ['Wz', 'wz'],
            'VelocityR': ['Vr', 'vr', 'r_velocity'],
            'VelocityTheta': ['Vt', 'vt', 'theta_velocity'],

            'ForceX': [''],
            'ForceY': [''],
            'ForceZ': ['Axial_Force', 'Axial Force'],
            'MomentumY': ['rov', 'roV', 'rovy', 'roVy', 'rhov', 'rhoVy', 'rhovy'],
            'MomentumZ': ['row', 'roW', 'rovz', 'roVz', 'rhow', 'rhoVz', 'rhovz'],
            'MomentumX': ['rou', 'roU', 'rovx', 'roVx', 'rhou', 'rhoVx', 'rhovx'],
            'RotatingMomentumY': ['rhoWy', 'rhowy', 'roWy', 'rowy'],
            'RotatingMomentumZ': ['rhoWz', 'rhowz', 'roWz', 'rowz'],
            'RotatingMomentumX': ['rhoWx', 'rhowx', 'roWx', 'rowx'],
            'MomentumXFlux': ['flux_rou'],
        	'MomentumYFlux': ['flux_rov'],
        	'MomentumZFlux': ['flux_row'],

            'Mach': ['Ma', 'mach', 'M', 'Mach_number'],  #:Mach number
            'RotatingMach': ['Mr'],  #:Mach number


            'VorticityX': ['vorticity_x', 'Vorticity_CoordinateX'],
	        'VorticityY': ['vorticity_y', 'Vorticity_CoordinateY'],
            'VorticityZ': ['vorticity_z', 'Vorticity_CoordinateZ'],
            'VorticityMagnitude': ['vorticity_modulus'],

            'Pressure': ['p', 'P', 'Ps', 'Psta', 'psta', 'ps', 'pressure', 'scalarPressure'], #:Static pressure
            'PressureStagnation': ['Pi', 'pgen', 'Pta'],  #:Stagnation pressure
            'RotatingPressureStagnation': ['Ptr'],  #:Stagnation pressure
            'Temperature': ['ts', 'tsta', 'Tsta', 'temperature', 'T', 'Ts'], #:Static temperature
            'TemperatureStagnation': ['Ti', 'tgen', 'Tta'],  #:Stagnation temperature

            'EnergyStagnation': ['E_a'],
            'EnergyStagnationDensity': ['roEa', 'rhoEa'],
            'Enthalpy': ['Hsta', 'hi'],
            'EnthalpyStagnation': ['Hta', 'stagnation_enthalpy'],  #:Stagnation enthalpy per unit mass
            'RotatingEnergyStagnation': ['E_r'],
            'RotatingEnergyStagnationDensity': ['roE', 'rhoE'],
            'RotatingEnthalpyStagnation': ['Htr'],  #:Stagnation enthalpy per unit mass

            'EnergyKinetic': ['Ec'],
            'Entropy': ['entropy', 's'],
            'TurbulentEnergyKinetic': ['k', 'tke', 'TKE'],
            'TurbulentEnergyKineticDensity': ['rok'],
            'TurbulentLengthScaleDensity': ['rok'],
            'TurbulentDissipation' : ['eps'],
            'ViscosityEddy': ['viscturb'],
            'Viscosity_EddyMolecularRatio' : ['viscrapp'],

            'IdealGasConstant': ['Rgaz', 'r_gaz', 'R_melange'],  #:Ideal gas constant (R = cp - cv)
            'VelocitySound': ['c', 'soundspeed', 'a'],  #:Static speed of sound
            'SpecificHeatRatio': ['gamma', 'Gamma', 'SpecificHeatRatio'],  #:Specific heat ratio
            'SpecificHeatPressure': ['Cp', 'cp'],  #:Specific heat at constant pressure
            'SpecificHeatVolume': ['Cv', 'cv'],  #:Specific heat at constant volume
            'Density': ['ro', 'rho', 'density']
        }

# Package-level docstring
__all__ = [
    'DEFAULT_NAME',
    'DEFAULT_SIMULATION_TYPE',
    'DEFAULT_SIMULATION_SETTINGS',
    'DEFAULT_ZONE_NAME',
    'DEFAULT_INSTANT_NAME',
    'DEFAULT_VERBOSE',
    'DEFAULT_VAR_SYNONYMS'
    # Add other module-level attributes or functions you want to expose
]

# Optional: Import statements for package modules or initialization code

## Import default Orion modules - This allow to do Orion.Base() for instance
from .Base import *
from .Reader import *
from .Writer import *
from .Ploter import *
from .ScriptParser import *
from .Formulas import *
from .SharedMethods import *

# Calculate the width of the header
header_width = 56

# Center the title within the header width
project_title = f"{__title__} Project"
centered_title = project_title.center(header_width)

# Header to print when the package is imported
header = f"""
{'=' * header_width}
{centered_title}
{'=' * header_width}
Author: {__author__}
Version: {__version__}
{'=' * header_width}
"""
print(header)
