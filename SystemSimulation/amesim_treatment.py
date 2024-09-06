#Python modules 
import sys
import os 

amesim_python_lib = r"C:\__BWI_appl__\AMEsim\2021.2\Amesim"
sub_paths = ["win64", r"scripting\python", r"scripting\python\win64"]
for sub_path in sub_paths: 
    sys.path.append(os.path.join(amesim_python_lib, sub_path))

try:
    import amesim_utils
    import amesim
    import ame_apy
except ModuleNotFoundError as err_msg:
    raise ModuleNotFoundError(err_msg)

ame_apy.AMEInitAPI()