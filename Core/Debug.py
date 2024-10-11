# Path and Char manipulation
import time
import psutil
import os
import platform
import threading
import sys
from multiprocessing import Manager
import threading

# Tools
import logging

# Orion
import Core as Orion
from SharedMethods import SharedMethods

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

class PerformanceStats(SharedMethods):
    """
    Context manager for collecting and printing performance statistics of a code block.

    This class measures various performance metrics such as execution time, CPU usage, 
    memory usage, I/O operations, and network activity during the execution of a code 
    block. It utilizes the `psutil` library to gather system-level information.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    The `__enter__` method starts the timer and records initial resource usage when 
    entering the context. The `__exit__` method collects the statistics upon exiting 
    the context and prints them out.

    The following performance metrics are collected:
    - Execution time: The total time taken for the code block to execute.
    - CPU usage: The percentage of CPU utilized during execution.
    - Memory usage: The difference in memory usage before and after execution, reported in MB.
    - I/O operations: Counts and bytes of read and write operations performed.
    - Network I/O: Bytes and packets sent and received over the network.
    - System metrics: Number of active processes and threads, disk usage statistics, and 
    system information including OS and Python version.

    Examples
    --------
    >>> with PerformanceStats() as stats:
    ...     # Code block to be measured
    ...     perform_heavy_computation()

    The statistics will be printed automatically upon exiting the `with` block.
    """
    def __init__(self):
        self.manager = Manager()
        self.stats = self.manager.dict()
        self.start_time = None
        self.process = psutil.Process()

    def __enter__(self):
        self.start_time = time.perf_counter()  # Use higher-precision time
        self.start_cpu_time = self.process.cpu_times()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # Memory in MB
        self.start_io = self.process.io_counters()
        self.start_net_io = psutil.net_io_counters()  # Start network counters
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._collect_stats()
        self.print_stats()

    def _collect_stats(self):
        end_time = time.perf_counter()  # Higher-precision end time
        end_cpu_time = self.process.cpu_times()
        end_memory = self.process.memory_info().rss / 1024 / 1024  # Memory in MB
        end_io = self.process.io_counters()
        end_net_io = psutil.net_io_counters()  # End network counters

        # Subtract start from end to get the delta for the monitored block
        self.stats['execution_time'] = end_time - self.start_time
        self.stats['cpu_usage_percent'] = psutil.cpu_percent(interval=0.1)
        self.stats['memory_usage'] = end_memory - self.start_memory
        self.stats['system_cpu_times'] = {
            'user': end_cpu_time.user - self.start_cpu_time.user,
            'system': end_cpu_time.system - self.start_cpu_time.system,
        }
        self.stats['io_counters'] = {
            'read_count': end_io.read_count - self.start_io.read_count,
            'write_count': end_io.write_count - self.start_io.write_count,
            'read_bytes': end_io.read_bytes - self.start_io.read_bytes,
            'write_bytes': end_io.write_bytes - self.start_io.write_bytes,
        }
        # Subtract start and end for network activity
        self.stats['network_io'] = {
            'bytes_sent': end_net_io.bytes_sent - self.start_net_io.bytes_sent,
            'bytes_recv': end_net_io.bytes_recv - self.start_net_io.bytes_recv,
            'packets_sent': end_net_io.packets_sent - self.start_net_io.packets_sent,
            'packets_recv': end_net_io.packets_recv - self.start_net_io.packets_recv,
        }

    def print_stats(self):
        self.print_text("info", "Performance Statistics:")
        print(f"Execution time: {self.stats['execution_time']:.2f} seconds")
        print(f"CPU Usage: {self.stats['cpu_usage_percent']:.2f}%")
        print(f"Memory Usage: {self.stats['memory_usage']:.2f} MB")
        print(f"System CPU Times - User: {self.stats['system_cpu_times']['user']:.2f}s, System: {self.stats['system_cpu_times']['system']:.2f}s")
        print(f"I/O Operations - Read: {self.stats['io_counters']['read_count']}, Write: {self.stats['io_counters']['write_count']}")
        print(f"I/O Bytes - Read: {self.stats['io_counters']['read_bytes'] / 1024:.2f} KB, Write: {self.stats['io_counters']['write_bytes'] / 1024:.2f} KB")
        
        self.print_text("info", "\nNetwork Statistics:")
        print(f"Network I/O - Bytes Sent: {self.stats['network_io']['bytes_sent'] / 1024:.2f} KB, Bytes Received: {self.stats['network_io']['bytes_recv'] / 1024:.2f} KB")
        print(f"Network Packets - Sent: {self.stats['network_io']['packets_sent']}, Received: {self.stats['network_io']['packets_recv']}")
        
        self.print_text("info", "\nMachine Statistics:")
        print(f"Number of Processes: {psutil.pids().__len__()}")
        print(f"Number of Threads: {threading.active_count()}")
        print(f"Disk Usage - Total: {psutil.disk_usage('/').total / (1024 ** 3):.2f} GB, Used: {psutil.disk_usage('/').used / (1024 ** 3):.2f} GB, Free: {psutil.disk_usage('/').free / (1024 ** 3):.2f} GB")
        print(f"System Info - OS: {platform.system()}, Python: {platform.python_version()}, CPU Count: {os.cpu_count()}")