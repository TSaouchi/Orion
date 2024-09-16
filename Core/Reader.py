
# Path and Char manipulation
import os
import sys
import pathlib
import re

# Tools
from collections import deque

# Data processing
import numpy as np
import scipy as spy
import pandas as pd

# I/O
import io
import csv
import h5py as hdf

# Orion
import Core as Orion
from SharedMethods import SharedMethods

# Message mode
import warnings
Verbose = Orion.DEFAULT_VERBOSE
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['VTK_SILENCE_GET_VOID_POINTER_WARNINGS'] = '1'

class Reader(SharedMethods):
    """
    Initialize the Reader object with the specified path and patterns.

    :ivar str path:
        The path to the directory containing the files.
    :ivar list patterns:
        The list of file patterns to be matched.
    """
    def __init__(self, path, patterns):
        """
        Initialize the Reader object with the specified path and patterns.

        :param path: The path to the directory containing the files.
        :type path: str

        :param patterns: The list of file patterns to be matched.
        :type patterns: list of str
        """
        super().__init__()
        self.path = self.path_manage(path)
        self.patterns = patterns

    def read_ascii(self, variables = None, file_selection = None,
                   zone_name = Orion.DEFAULT_ZONE_NAME,
                   instant_name = Orion.DEFAULT_INSTANT_NAME, **kwargs):

        files_dict = self.__generate_file_list(self.path, self.patterns)

        # Selecting a file is just slicing in the files dictionary
        if file_selection is not None:
            files_dict = self.__slice_dictionary(files_dict, file_selection)

        # If file not file found stop the code
        self.__file_not_found(files_dict, self.path, self.patterns)
        # If a single file is read, by default the instant name is Default else one can give a name
        instant_naming = self.__instant_naming(instant_name[0], files_dict)

        base = Orion.Base()
        base.init(zone_name, [instant_naming(value) for _, value in files_dict.items()])

        self.print_text("info", f"\nReading input files")
        for nfile, file in enumerate(files_dict):
            self.print_text('check', f"\n\t{file}")
            file_path = os.path.join(self.path, file)

            custom_header = kwargs.get('custom_header', None)
            verbose = kwargs.get('verbose', Verbose)

            _, file_data = self.__read_ascii(file_path, variables, custom_header, verbose)

            # Create the base
            for variable in file_data:
                base[zone_name[0]][nfile].add_variable(self._remove_spaces(variable),
                                                    pd.to_numeric(file_data[variable],
                                                                errors='coerce'
                                                                ).dropna().to_numpy())
        return self.variable_mapping_cgns(base)

    def read_mat(self, variables = None, file_selection = None, zone_name = Orion.DEFAULT_ZONE_NAME,
                   instant_name = Orion.DEFAULT_INSTANT_NAME):

        if variables is None:
            self.print_text("error", "We hate none HDF5 .mat files structure, thus give us a sequence of variables to read")
            raise ValueError

        files_dict = self.__generate_file_list(self.path, self.patterns)

        # Selecting a file is just slicing in the files dictionary
        if file_selection is not None:
            files_dict = self.__slice_dictionary(files_dict, file_selection)

        # If file not file found stop the code
        self.__file_not_found(files_dict, self.path, self.patterns)
        # If a single file is read, by default the instant name is Default else one can give a name
        instant_naming = self.__instant_naming(instant_name[0], files_dict)

        base = Orion.Base()
        base.init(zone_name, [instant_naming(value) for _, value in files_dict.items()])

        self.print_text("info", f"\nReading input files")
        for nfile, file in enumerate(files_dict):
            self.print_text('check', f"\n\t{file}")
            file_path = os.path.join(self.path, file)

            file_data = {}
            spy.io.loadmat(file_path, mdict = file_data,
                           variable_names = variables)

            # Create the base
            for variable in variables:
                base[zone_name[0]][nfile].add_variable(self._remove_spaces(variable),
                                                       np.squeeze(file_data[variable]))
        return self.variable_mapping_cgns(base)

    def _remove_spaces(self, variable):
        return re.sub(r'(?<=\S)\s+(?=\S)', '_', variable.strip())

    def __slice_dictionary(self, dictionary, file_selection):
        if np.min(file_selection) < 1:
            self.print_text("error", "File selection must be greater than or equal to one.")
            raise ValueError
        min_value = np.min(file_selection)
        max_value = np.max(file_selection)
        dictionary = {key: value for index,
                          (key, value) in enumerate(dictionary.items(),
                                                    start=1) if index >= min_value and index <= max_value}

        return dictionary

    def __instant_naming(self, instant_name, files_dict):
        if len(files_dict) == 1:
            instant_naming = lambda x : str(instant_name)
        else:
            instant_naming = lambda x : f"inst{'%04d' % x}"
        return instant_naming

    def __read_ascii(self, file_path, variables = None, custom_header = None, verbose = True, **kwargs):
        # Determine the file extension
        file_extension = file_path.split('.')[-1].lower()

        # Read the file content
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Detect the delimiter
        if file_extension == 'csv':
            delimiter = kwargs.get("CSV_delimiter", '\t')
        else:  # For .dat, .txt, and other flat files, we'll auto-detect
            first_data_line = next((line for line in lines if re.match(r'^[\w\s.,-]+([,;\t][\w\s.,-]+)+$', line)), None)
            if first_data_line:
                if '\t' in first_data_line:
                    delimiter = '\t'
                elif ';' in first_data_line:
                    delimiter = ';'
                else:
                    delimiter = ','
            else:
                raise ValueError("Unable to detect delimiter in the file")

        if custom_header is None:
            # Extract header information
            header_info = []
            data_start = 0
            for i, line in enumerate(lines):
                if re.match(f'^[\\w\\s.,-]+({delimiter}[\\w\\s.,-]+)+$', line):
                    data_start = i
                    break
                header_info.append(line.strip())

            # Extract column names from the file
            column_names = lines[data_start].strip().split(delimiter)
            clean_columns = [re.sub(r'\s*\([^)]*\)$', '', col.strip()) for col in column_names]

            # Join the data lines into a single string
            data_content = ''.join(lines[data_start + 1:])
        else:
            # If custom header is provided, skip all string content
            header_info = None
            data_content = ''
            for line in lines:
                if re.match(f'^[\\d.,-]+({delimiter}[\\d.,-]+)*$', line):
                    data_content += line

            clean_columns = custom_header

        if variables is not None and custom_header is None:
            clean_columns = variables

        # Use pandas to read the data content
        df = pd.read_csv(io.StringIO(data_content), sep=delimiter,
                         names=clean_columns, usecols=variables,
                         skipinitialspace=True)
        if verbose:
            if header_info:
                print("Header Information:")
                for line in header_info:
                    print(line)

            print(f"\nDetected Delimiter: '{delimiter}'")
            print("\nData Preview:")
            print(df.head())
            print("\nColumn Names:")
            print(df.columns.tolist())
            print("\nData Types:")
            print(df.dtypes)

        return header_info, df

    def read_cgns(self, file_selection = None, instant_name = '0000',
                  path_to_mesh = None, enable_pycgns = False):

        path = self.path
        patterns = self.patterns
        cgns_files = self.__generate_file_list(path, patterns)

        if enable_pycgns:
            enable_pycgns = self.is_pycgns_available()

        # Selecting a file is just slicing in the files dictionary
        if file_selection is not None:
            if np.min(file_selection) < 1:
                self.print_text("error", "File selection must be greater than or equal to one.")
                raise ValueError
            min_value = np.min(file_selection)
            max_value = np.max(file_selection)
            cgns_files = {key: value for index, (key, value) in enumerate(cgns_files.items(), start=1) if index >= min_value and index <= max_value}

        # If file not file found stop the code
        self.__file_not_found(cgns_files, path, patterns)
        # If a single file is read, by default the instant name is '0000' else one can give a name
        if len(cgns_files) == 1:
            instant_naming = lambda x : str(instant_name)
        else:
            instant_naming = lambda x : f"inst{'%04d' % x}"

        self.print_text("info", f"\nReading input files")

        if enable_pycgns:
            self.print_text("info", f"\tEnable PyCNGS : {enable_pycgns}")
            import CGNS.PAT.cgnskeywords as _ck
            import CGNS.PAT.cgnsutils as _cu
            import CGNS.MAP
            reader = pyvista.Reader('pycgns')
        else:
            reader = pyvista.Reader('hdf_cgns')

        if path_to_mesh is None:
            for nfile, (file, value) in enumerate(cgns_files.items()):
                self.print_text('check', f"\n\t{file}")
                if enable_pycgns:
                    tree, _, _ = CGNS.MAP.load(os.path.join(path, file))
                    cgns_base = _cu.hasChildType(tree, _ck.CGNSBase_ts)
                    reader['object'] = cgns_base[0]
                else:
                    reader['filename'] = os.path.join(path, file)

                if nfile == 0:
                    base = reader.read()
                    reader["base"] = base
                else:
                    base = reader.read()
                for zone in base:
                    base[zone][instant_naming(value)] = base[zone].pop(nfile)

        elif path_to_mesh is not None:
            #!  keep it this way in case this has to evolve to read a mesh by instant
            # check if the mesh exist - if not stop execution
            self.__file_not_found(os.path.exists(path_to_mesh),
                                  path_to_mesh,
                                  os.path.basename(path_to_mesh))
            #:Read the mesh first
            self.print_text('check', f"\tRead mesh file first: {os.path.basename(path_to_mesh)}")
            if enable_pycgns:
                tree, _, _ = CGNS.MAP.load(path_to_mesh)
                cgns_base = _cu.hasChildType(tree, _ck.CGNSBase_ts)
                reader['object'] = cgns_base[0]
            else:
                reader['filename'] = path_to_mesh

            base = reader.read()

            for nfile, (file, value) in enumerate(cgns_files.items()):
                self.print_text('check', f"\n\t{file}")
                if enable_pycgns:
                    tree, _, _ = CGNS.MAP.load(os.path.join(path, file))
                    cgns_base = _cu.hasChildType(tree, _ck.CGNSBase_ts)
                    reader['object'] = cgns_base[0]
                else:
                    reader['filename'] = os.path.join(path, file)

                if nfile == 0:
                    reader["base"] = base

                base = reader.read()
                for zone in base:
                    base[zone][instant_naming(value)] = base[zone].pop(nfile)

        base = self.variable_mapping_cgns(base)
        return base

    def read_csv(self, separator = None, shared = False):
        """
        Read csv input files and return a dictionary.

        :param separator: The separator used in the csv file. Default is None, which sets it to comma.
        :type separator: str or None
        :param shared: Whether to share coordinate values among all instances. Default is False.
        :type shared: bool

        :return: A dictionary containing the data read from the csv file.
        :rtype: dict

        .. note::

            - File structure of the output file is:

                ``index`` ``instant`` ``<Coordinates>`` ``<Variables>``
        """
        path = self.path
        patterns = self.patterns
        if isinstance(patterns, list):
            patterns = patterns[0]

        file_path = os.path.join(path, patterns)

        # If not file found stop the code
        self.__file_not_found(os.path.exists(file_path), path, patterns)


        if separator is None:
            # Default separator is space
            separator = ','

        self.print_text("info", "\nReading input file")
        self.print_text('check', f"\n\t{patterns}")

        data_dict = {}
        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter = separator)
            header = next(reader)  # Read header
            if shared:
                # If shared=True, create a 'Coordinate' dictionary to hold coordinate values
                # Assuming the first three variables are coordinates
                coordinate_names = header[2:5]
                # Read the next row to get coordinate values
                coordinate_values = next(reader)[2:5]
                data_dict['Coordinate'] = {name: self.__convert_to_type(value) for name, value in zip(coordinate_names, coordinate_values)}
                csvfile.seek(0)
                reader = csv.reader(csvfile, delimiter = separator)
                header = next(reader)  # Read header
            for row in reader:
                # Assuming the instant is in the second column
                instant = row[1]
                if instant not in data_dict:
                    data_dict[instant] = {}
                # Assuming variables start from the third column
                for var_index, var_name in enumerate(header[2:]):
                    # Exclude coordinate variables if shared=True
                    if not (shared and var_name in coordinate_names):
                        # +2 to skip index and instant columns
                        data_dict[instant][var_name] = self.__convert_to_type(row[var_index + 2])

        return data_dict

    def __convert_to_type(self, value):
        """
        Function to convert list elements to appropriate data types
        """
        try:
            return int(value)  # Try converting to integer
        except ValueError:
            try:
                return float(value)  # Try converting to float
            except ValueError:
                try:
                    return eval(value)  # Try evaluating as Python expression
                except:
                    return value  # Return as string if all conversion attempts fail

    def __generate_file_list(self, path, patterns):
        """
        Generate a dictionary of files matching the specified patterns.

        :param path: The path where files are located.
        :type path: str

        :param patterns: The file patterns to match. If a list is provided, only the first pattern will be used.
        :type patterns: str or list

        :return: A dictionary containing filenames as keys and their corresponding iteration numbers as values or index.
        :rtype: dict

        :raises ValueError: If no files are found matching the specified patterns.

        .. note::
            - If '<instant>' is present in patterns, the iteration are detected.
            - If '*' is include in the pattern it will be used as a wildcard to match any character.
            - Filenames are sorted based on iteration numbers in ascending order.
        """

        if isinstance(patterns, list):
            patterns = patterns[0]

        # Use pathlib to handel very long paths
        path_obj = pathlib.Path(path)

        if '<instant>' in patterns:
            # Find all files matching the full pattern
            files = list(path_obj.glob(patterns.replace('<instant>', '*')))
            self.__file_not_found(files, path, patterns)

            # Extract iteration numbers from filenames
            file_list = {}
            for file in files:
                files_basename = os.path.basename(file)
                # Construct the regex pattern with the correct instant placeholder and escape special caracters
                regex_pattern = re.escape(patterns).replace('<instant>', r'(\d+)')
                match = re.search(regex_pattern, files_basename)
                if match:
                    file_list[files_basename] = int(match.group(1))

            # Sort according to iterations numbers (jsut in case)
            file_list = dict(sorted(file_list.items(),
                               key=lambda item: item[1],
                               reverse = False))
            return file_list
        else:
            file_list = list(path_obj.glob(patterns))
            self.__file_not_found(file_list, path, patterns)
            file_list  = [os.path.basename(file) for file in file_list]
            file_list.sort()
            file_list = {os.path.basename(file): index for index, file in enumerate(file_list)}
            return file_list

    def __file_not_found(self, files, path, patterns):
        """
        Check if files matching the specified patterns are found. If not, raise an error and exit the program.

        :param files: List of files found matching the patterns.
        :type files: list

        :param path: The path where the search was performed.
        :type path: str

        :param patterns: The file patterns used for the search.
        :type patterns: str

        :raises SystemExit: If no files are found matching the specified patterns.

        .. note::
            - This function is called internally by __generate_file_list.
            - It prints an error message if no files are found and exits the program.
        """
        if not files:
            error_message = f"Error: no files found with the given patterns: {patterns}"
            info_message = f"Target path: {path}\n"
            self.print_text("error", error_message)
            self.print_text("info", info_message)
            sys.exit()