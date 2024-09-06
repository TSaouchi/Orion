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
    #! import the logger
    #! Keep it simple and stupid for now - each debugging method is independent and redundant
    def __init__(self, loggername = "debug.log"):
        if debug:
            self.loggername = loggername

    def memory_usage(self):
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