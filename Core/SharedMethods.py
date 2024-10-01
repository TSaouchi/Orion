# Path and Char manipulation
import os
import re
from collections import defaultdict
from tqdm import tqdm

# Data processing
import numpy as np

from Core import DEFAULT_VAR_SYNONYMS

class SharedMethods():
    def __init__(self):
        pass

    def path_manage(self, path):
        """
        Modify the given path based on the operating system.

        :param path: The path to be modified.
        :type path: str

        :return: The modified path.
        :rtype: str

        .. note::
            - If used, the input path should always be Windows path format
            - If the operating system is Windows (nt), the function replaces occurrences of ``\\\\data`` at the beginning
            of the path with ``\\\\?\\UNC\\data``.
            - If the operating system is POSIX (posix), the function normalizes the path, converts backslashes to
            forward slashes, and returns the result.
        """
        if os.name == 'nt':
            unc_pattern = re.compile(r'^\\\\')
            if bool(unc_pattern.match(path)):
                return path.replace(r"\\", '\\\\?\\UNC\\')
            else:
                return '\\\\?\\' + path

        if os.name == 'posix':
            return os.path.normpath(path).replace("\\\\", "/").replace("\\", "/")

    def export_path_check(self):
        """
        Check and create the export directory.

        Returns
        -------
        str
            The path to the export directory.

        :raises: Any exceptions that may occur during the process.

        .. note::
            - This method checks if the export directory exists. If not, it creates the directory.
            - The export directory is determined by combining the specified path and the directory name tag.
            - The method ensures that the export directory is created before returning the path.

        """
        path = self.path
        export_path = os.path.join(path, f"{self.dir_name_tag}")
        if not os.path.isdir(export_path):
            os.makedirs(export_path)
        return export_path

    def variable_mapping_cgns(self, base):
        """
        - Map variable names in the given dataset to standardized names.
        - CGNS nomenclature:
            - https://cgns.github.io/CGNS_docs_current/sids/dataname.html

        :param base: The input dataset.
        :type base: Dataset

        :return: The dataset with standardized variable names.
        :rtype: Dataset

        .. note::
            - The function uses a dictionary `VAR_SYNONYMS` to map variable names to standardized names.
            - If a variable in the dataset does not have a standardized name in the dictionary, it attempts to find a match based on synonyms defined in `VAR_SYNONYMS`.
            - If a variable is not found in the synonyms, it remains unchanged.
            - The function then renames variables in the dataset according to the mapping defined.
        """
        
       
        VAR_SYNONYMS = DEFAULT_VAR_SYNONYMS
        all_base_variables = self.variables_location(base)
        var_to_rename = list(set(all_base_variables) - set(list(VAR_SYNONYMS.keys())))
        mapping_var = {item: key for item in var_to_rename for key, synonyms in VAR_SYNONYMS.items() if item in synonyms}
        try:
            base.rename_variable(list(mapping_var), list(mapping_var.values()))
        except:
            pass
        
        return base
    
    def variables_location(self, base, location = False):
        variable_map = defaultdict(list)

        # Build the variable map by checking all zones and instants
        for zone_name, zone in base.zones.items():
            for instant_name, instant in zone.instants.items():
                for variable_name,_ in instant.variables.items():
                    variable_map[variable_name].append((zone_name, instant_name))
        if location:
            return variable_map
        
        return list(variable_map.keys())

    def print_table(self, table):
        """
        Print a formatted table with keys and values.

        :param table: A dictionary representing the table with keys and values.
        :type table: dict

        :return: None

        .. note::
            - This method prints a formatted table where keys and values are displayed in rows.
            - The maximum length of keys is determined for proper formatting.
            - If a value is a list, it is formatted as a comma-separated string.
        """
        #:Find the maximum length of keys for formatting
        max_key_length = np.max(map(len, table.keys()))

        #:Print the table
        for key, value in table.items():
            #:Check if the value is a list and format it accordingly
            if isinstance(value, list):
                value = ', '.join(map(str, value))

            #:Format and print each row
            print(f"{key.ljust(max_key_length)} : {value}")

    def print_text(self, level, text):
        """
        Print colored text based on the specified logging level.

        :param level: The logging level (e.g., 'error', 'check', 'info', 'warning').
        :type level: str
        :param text: The text to be printed.
        :type text: str

        :return: None

        :raises ValueError: If an invalid logging level is provided.

        .. note::
            - The function uses ANSI escape codes for coloring the text.
            - Valid logging levels and their corresponding colors:
                - 'error': Red
                - 'check': Green
                - 'info': Blue
                - 'warning': Orange (added for 'warning' level)
            - If an invalid logging level is provided, a ValueError is raised, and the text is printed without color.
            - The function automatically resets the text color to default after printing.
        """
        colors = {
            'error': "\x1B[31m",     # Red
            'check': "\x1B[32m",     # Green
            'info': "\x1B[34m",      # Blue
            'warning': "\x1B[33m",   # Yellow
            'text': "\x1B[37m",      # White
            'reset': "\x1B[0m",      # Reset to default color
            'black': "\x1B[30m",     # Black
            'cyan': "\x1B[36m",      # Cyan
            'magenta': "\x1B[35m",   # Magenta
            'red': "\x1B[91m",       # Bright Red
            'green': "\x1B[92m",     # Bright Green
            'yellow': "\x1B[93m",    # Bright Yellow
            'blue': "\x1B[94m",      # Bright Blue
            'magenta': "\x1B[95m",   # Bright Magenta
            'cyan': "\x1B[96m",      # Bright Cyan
            'light_gray': "\x1B[37m",# Light Gray
            'dark_gray': "\x1B[90m"  # Dark Gray
        }

        try:
            if level not in colors:
                raise ValueError(f"Invalid logging level: {level}")
            complet_text = colors[level] + text
            dotdot = (80 - len(text))*'~'
            if level == 'info':
                print(complet_text + dotdot + "\x1B[0m")
            else:
                print(complet_text + "\x1B[0m")

        except ValueError as e:
            print("Warning:", e)

    def tqdm_wrapper(self, iterable, desc=None, unit=None, verbose=False):
        """
        A generic wrapper for tqdm progress bar.

        Parameters:
        -----------
        iterable : iterable
            The iterable to wrap with tqdm.
        desc : str, optional
            Description to be displayed alongside the progress bar.
        unit : str, optional
            The unit of the items being iterated over.
        verbose : bool, optional
            If True, use tqdm. If False, return the original iterable. Default is False.

        Returns:
        --------
        iterable
            The original iterable wrapped with tqdm if verbose is True, otherwise the original iterable.
        """
        if verbose:
            unit = "ops" if  unit is None else unit
            return tqdm(iterable, desc=desc, unit=unit, unit_scale=True, 
                        ncols=80, colour="green")
        return iterable

    def is_pycgns_available(self):
        try:
            import CGNS.MAP
            return True
        except:
            return False