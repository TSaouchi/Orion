# Path and Char manipulation
import os
import re

# Tools
from collections import deque

# Data processing
import numpy as np

# I/O
import io
import csv
import h5py as hdf

# Orion
import Core as Orion
from SharedMethods import SharedMethods

# Message mode
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['VTK_SILENCE_GET_VOID_POINTER_WARNINGS'] = '1'

class Writer(SharedMethods):
    """
    Class to export and write data to CGNS and XMF files.
    """
    def __init__(self, base, path, dir_name_tag = "output", files_name_tag = "output_result"):
        """
        Initialize the object.

        Parameters
        ----------
        base : pyvista.Base
            The pyvista.Base object containing the data to be processed.
        path : str
            The path to the directory where the output will be exported.
        dir_name_tag : str, optional
            Tag used in the directory name for export, by default "output".
        files_name_tag : str, optional
            Tag used in the file names for export, by default "output_result".

        :raises: Any exceptions that may occur during the process.

        .. note::
            - This method initializes the object with the provided pyvista.Base object (`base`),
            export directory path (`path`), and optional directory and file name tags.
            - The default directory name tag is "output", and the default file name tag is "output_result".
            - The initialized object is ready to perform operations such as writing CGNS files based on
            the specified target row.

        """
        self.dir_name_tag = dir_name_tag
        self.files_name_tag = files_name_tag
        self.base, self.path = base, path

    def write_cgns(self, link_cgns_files = None, save_mesh_at_instant = None,
                   moving_mesh = False, enable_pycgns = False):
        """
        Write CGNS files

        :param link_cgns_files:
            Type of linking to use for CGNS files. Possible values:
                - None: Don't share the mesh use the default writing mode.
                - `mesh_to_solution`: Share the mesh and link the mesh file to the solution files (each solution file is mester and only the mesh file is slave).
                - `solution_to_mesh`: Share the mesh and link the solution files to the mesh file (mesh file is master, solutions files are slaves).
        :type link_cgns_files: str, optional

        :param save_mesh_at_instant: If True, save the mesh at a given instant.
        :type save_mesh_at_instant: bool, optional

        :moving_mesh: if True, all instant are interpolated on the first mesh.
        :type moving_mesh: bool, default False

        :param path_to_mesh: If one want to read using the module pycngs
        :type path_to_mesh: bool, optional

        :return: None

        :raises: Any exceptions that may occur during the process.

        .. note::
            - This method writes CGNS files based on the specified target row. It uses the pyvista library for merging and writing CGNS files.
            - If `merge_before_writing` is set to True, the data will be merged using the "merge" treatment from pyvista before writing the CGNS files.
            - The method determines the export path using the `__export_path_check` method.
            - If `link_cgns_files` is None, the method writes separate CGNS files for each instant without sharing the mesh.
            - If `link_cgns_files` is "mesh_to_solution", the method links the mesh file to the solution files.
            - If `link_cgns_files` is "solution_to_mesh", the method links the solution files to the mesh file.
            - If `save_mesh_at_instant` is True, the method saves the mesh at the specified instant.

                - This parameter has sens only if link_cgns_files is used
                - This parameter is Boolean if ``True`` at a given instant, then the saved mesh is that of that instant. If ``None``, the default mesh is that of the last instant.
                - In case of only one IBI we don't need to take account of the time step
                - If the input base of the reconstruction contains all instants, the first instant will be saved by default (can be modified if necessary).
                e.g: how to use it ? ``save_mesh_at_instant = instant == list_of_instant[index]``
        """
        #! If base is empty pass this method
        if not self.__check_base(self.base): return
        #! Retrieve the base after writing
        base = self.base if link_cgns_files is None else copy.deepcopy(self.base)
        export_path = self.export_path_check()
        if enable_pycgns:
            enable_pycgns = self.is_pycgns_available()

        # Instant naming
        if len(base[0].keys()) == 1 and base[0].keys() == ['0000']:
            file_naming = lambda instant: os.path.join(export_path, f'{self.files_name_tag}.cgns')
        else:
            file_naming = lambda instant: os.path.join(export_path, f'{self.files_name_tag}_{instant}.cgns')

        if enable_pycgns:
            writer = pyvista.Writer('pycgns')
        else:
            writer = pyvista.Writer('hdf_cgns')
        # Don't share the mesh and use this as default writing mode
        if link_cgns_files is None:
            self.print_text("info", "\nWriting output files")
            for instant in base[0].keys():
                writer['filename'] = file_naming(instant)
                #WARN - Slicing is fast but does not ensure inheretence of bcs conectivity
                writer['base'] = base[:, (instant,)]
                writer.dump()
        else:
            # Share the mesh and link the mesh file to the solution files
            #Or,
            # Share the mesh and link the solution files to the mesh file
            self.__link(base,
                        link_type = link_cgns_files,
                        save_mesh_at_instant = save_mesh_at_instant,
                        moving_mesh = moving_mesh,
                        enable_pycgns = False)

    def write_csv(self, separator = None, coordinates = None):
        """
        Write CSV files.

        :param separator: The separator to be used in the CSV file. Defaults to None.
        :type separator: str, optional

        .. note::
            - The first line contains the name of variables separated by the separator string. Columns may have different sizes. Rows are composed of numbers separated by the separator string.
            - This is an Instant writer (It only writes an Instant in a file). So, if many zones are in a base, then it will write many files if the tag ``<zone>`` is in the ``filename``. Otherwise, the zone name is appended at the end of the ``filename``.

            - Possible seperators:

                - Comma (','): The most common choice - Default.
                - Space (' '): space separator.
                - Tab ('\t'): Used for TSV (Tab-Separated Values) files.
                - Semicolon (';'): Alternative in some regions.
                - Pipe ('|'): Occasional choice for data files.
                - Colon (':'): Sometimes used, especially in configurations.
                - Slash ('/'): Used in paths or URLs.
                - Underscore ('_'): Occasionally used.
                - Dash ('-'): Sometimes used for dates or numeric ranges.
                - Caret ('^'): Less common but possible.

            - File structure of the output file is:

                ``index`` ``instant`` ``<Coordinates>`` ``<Variables>``
        """
        if not self.__check_base(self.base): return
        base = self.base
        self.project_solution(base, location = 'node', reset = True)
        export_path = self.export_path_check()
        path = lambda x: os.path.join(export_path, f'{self.files_name_tag}_{x}.csv')

        if separator is None:
            # Default separator is space
            separator = ','

        if coordinates is None:
            coordinates = base.coordinate_names
        else:
            coordinates += base.coordinate_names

        self.print_text("info", "\nWriting output files")

        variables = coordinates + list(set(self.base_variables_names(base)) - set(coordinates))
        # Iterate over each zone
        for zone in progress_bar(base, label = 'Zone'):
            with open(path(zone), 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter = separator)

                # Write header
                header = ['index', 'instant'] + variables
                writer.writerow(header)

                # Write data rows
                for ninstant, instant in enumerate(base[zone]):
                    # Take only the digits
                    if 'inst' in instant:
                        instant_digit = re.sub(r'[^0-9]', '', instant)
                    else:
                        instant_digit = instant
                    values = [ninstant, instant_digit] + [self.__convert_to_list(base[zone][instant][var]) for var in variables]
                    writer.writerow(values)

    def __convert_to_list(self, value):
        if isinstance(value, np.ndarray) and len(value) > 1:
            return value.tolist()
        else:
            return value[0]