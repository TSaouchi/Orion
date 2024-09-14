# Path and Char manipulation
import os
import sys

# Tools
import logging

# Orion
import Core as Orion

# Debug script manually
import pdb
Verbose = Orion.DEFAULT_VERBOSE
debug = "--debug" in sys.argv

class Debug():
    """
    A simple debugging utility class that provides methods for logging memory usage 
    and setting interactive breakpoints in the script when debugging is enabled.

    Attributes:
    -----------
    loggername : str
        The name of the log file where debug information is written.
    logger : logging.Logger
        Logger object for managing logging.
    handler : logging.FileHandler
        Handler for writing log messages to the file.
    formatter : logging.Formatter
        Formatter for formatting log messages.
    """
    #! import the logger
    #! Keep it simple and stupid for now - each debugging method is independent and redundant
    def __init__(self, loggername = "debug.log"):
        """
        Initialize the Debug class. The logger is only set up if the --debug flag is passed.

        Parameters:
        -----------
        loggername : str
            The name of the log file to store debug information. Defaults to 'debug.log'.
        """
        if debug:
            self.loggername = loggername

    def memory_usage(self):
        """
        Logs the memory usage of all global and local objects in the script when debugging is enabled.

        The method logs the memory usage in both GB and MB, detailing the memory consumption 
        for each object, as well as the total memory usage of the script. Uses the `pympler.asizeof` 
        to calculate memory sizes.
        """
        if debug:
            # Create a new logger object
            self.logger = logging.getLogger("Memory usage")
            self.handler = logging.FileHandler(self.loggername, mode = "w")
            self.formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
            self.handler.setFormatter(self.formatter)
            self.logger.addHandler(self.handler)
            self.logger.setLevel(logging.DEBUG)

            from pympler.asizeof import asizeof
            import types
            # Get all objects created in your code
            all_objects = {}
            for obj_name, obj in globals().items():
                if not obj_name.startswith("__") and not obj_name.startswith("_") and not isinstance(obj, types.ModuleType):
                    all_objects[obj_name] = obj
            for obj_name, obj in locals().items():
                if not obj_name.startswith("__") and not obj_name.startswith("_") and not isinstance(obj, types.ModuleType):
                    all_objects[obj_name] = obj

            total_memory_usage = 0
            for obj_name, obj in all_objects.items():
                memory_usage_bytes = asizeof(obj)
                total_memory_usage += memory_usage_bytes

                self.logger.debug(f"{obj_name}: {round(memory_usage_bytes / (1024 ** 3), 4)} GB, {round(memory_usage_bytes / (1024 ** 2), 4)} MB")

            self.logger.debug(f"\t\tTotal: {round(total_memory_usage / (1024 ** 3), 4)} GB, {round(total_memory_usage / (1024 ** 2), 4)} MB\n")

    def breakpoint(self):
        """
        Triggers an interactive Python Debugger (pdb) breakpoint when debugging is enabled.

        When called, it logs a message indicating that a breakpoint has been reached and 
        provides an interactive session in which the user can execute commands. The output
        of the session is written to the log file specified during initialization.
        """
        if debug:
            self.logger = logging.getLogger("breakpoint")
            self.handler = logging.FileHandler(self.loggername, mode = "w")
            self.logger.addHandler(self.handler)

            print(">>>>> breakpoint <<<<<")
            print(">> Start interactive session : outputs are redirected to :")
            print(f"\t{self.loggername}")
            print(">> Press: ")
            print("\t 'c' to continue")
            print("\t 'h' to list all commands")
            pdb.Pdb(stdout = self.handler.stream).set_trace()