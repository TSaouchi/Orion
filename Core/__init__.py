# Metadata
__title__ = 'Orion'
__description__ = 'A package for data processing, system simulation, and CFD post-processing.'
__version__ = 'X.X.X'
__author__ = 'Toufik Saouchi'
__author_email__ = 'toufik.saouchi@gmail.com'
__license__ = '[Your License]'
__github_url__='https://github.com/TSaouchi/Orion'

# Default variables
DEFAULT_NAME = 'Orion'
DEFAULT_VERBOSE = True
DEFAULT_DEBUG = False
DEFAULT_SIMULATION_TYPE = '1D'
DEFAULT_ZONE_NAME = ['Zone']
DEFAULT_INSTANT_NAME = ['Instant']
DEFAULT_SIMULATION_SETTINGS = {
    'tolerance': 1e-5,
    'max_iterations': 1000
}
DEFAULT_TIME_NAME = ['TimeValue']
DEFAULT_TIMESTEP_NAME = ['TimeStep']
DEFAULT_FREQUENCY_NAME = ['Frequency']
DEFAULT_COORDINATE_NAMES = ['CoordinateX','CoordinateY','CoordinateZ']

DEFAULT_VAR_SYNONYMS = {

            DEFAULT_TIME_NAME[0] : ['t', 'Temps', 'Time', 'time', 'Temps__1_-_Vitesse_de_mesure_standard', 'Temps__2_-_Vitesse_de_mesure_standard'],
            DEFAULT_TIMESTEP_NAME[0] : ['dt', 'Deltat', 'time_step'],
            DEFAULT_FREQUENCY_NAME[0] : ['frequency'],
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
            'Density': ['ro', 'rho', 'density'],
            
            # None CGNS nomenclature
            'Current' : ['Courant_1'],
            'PressureStagnationIn': ['Pressure_IN_V2_++'],
            'PressureStagnationOut': ['Pressure_OUT_V1_R'],
            'FlowRate' : ['Debit_Q3_V7_BHG']
        }

# Package-level docstring
__all__ = [
    'DEFAULT_NAME',
    'DEFAULT_SIMULATION_TYPE',
    'DEFAULT_SIMULATION_SETTINGS',
    'DEFAULT_ZONE_NAME',
    'DEFAULT_INSTANT_NAME',
    'DEFAULT_VERBOSE',
    'DEFAULT_DEBUG,'
    'DEFAULT_VAR_SYNONYMS',
    'DEFAULT_TIME_NAME',
    'DEFAULT_TIMESTEP_NAME',
    'DEFAULT_FREQUENCY_NAME',
    'EDFAULT_COORDINATE_NAMES'
    # Add other module-level attributes or functions you want to expose
]

# Optional: Import statements for package modules or initialization code

## Import default Orion modules - This allow to do Orion.Base() for instance
from .Base import *
from .Reader import *
from .Writer import *
from .Plotter import *
from .ScriptParser import *
from .Formulas import *
from .SharedMethods import *


# Header to print when the package is imported
import multiprocessing

header_width = 80  # Adjust as needed
centered_title = f"Project {__title__}".center(header_width)
header = f"""
{'=' * header_width}
{centered_title}
{'=' * header_width}
Author: {__author__}
Version: {__version__}
{'=' * header_width}
"""

def print_header_once():
    if multiprocessing.current_process().name == 'MainProcess':
        print(header)

# Call this function at the beginning of your script
print_header_once()
