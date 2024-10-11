import numpy as np
import dask.array as da
import re
from collections import OrderedDict, defaultdict

# For base computation level
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor

# Orion parameters
from Core import DEFAULT_ZONE_NAME, DEFAULT_INSTANT_NAME, DEFAULT_VERBOSE
from SharedMethods import SharedMethods

class CustomAttributes(SharedMethods):
    """
    Class to manage custom attributes in a structured way.
    
    Attributes
    ----------
    _attributes : OrderedDict
        Internal storage for custom attributes.
    """
    def __init__(self):
        """Initialize an empty attribute dictionary.""" 
        super().__init__()
        self._attributes = OrderedDict()

    def set_attribute(self, name, value):
        """
        Set or update an attribute.

        Parameters
        ----------
        name : str
            The name of the attribute.
        value : Any
            The value of the attribute.
        """
        self._attributes[name] = value

    def get_attribute(self, name = None, default = None):
        """
        Get the value of an attribute.

        Parameters
        ----------
        name : str
            The name of the attribute.
        default : Any, optional
            The default value to return if the attribute is not found.

        Returns
        -------
        Any
            The value of the attribute, or the default value if not found.
        """
        if not name:
            return self._attributes
        return self._attributes.get(name, default)

    def delete_attribute(self, name):
        """
        Delete an attribute.

        Parameters
        ----------
        name : str
            The name of the attribute to delete.
        """
        try:
            del self._attributes[name]
        except KeyError:
            raise KeyError(f"Attribute '{name}' does not exist.")

    def rename_attribute(self, old_name, new_name):
        """
        Rename an attribute.

        Parameters
        ----------
        old_name : str
            The current name of the attribute.
        new_name : str
            The new name of the attribute.
        """
        if old_name not in self._attributes:
            raise KeyError(f"Attribute '{old_name}' does not exist.")
        self._attributes[new_name] = self._attributes.pop(old_name)

class Variables(CustomAttributes):
    """
    Class representing a variable with data and attributes.
    
    Attributes
    ----------
    data : dask.array.Array
        The data associated with the variable.
    """
    def __init__(self, data):
        """
        Initialize the Variables object.

        Parameters
        ----------
        data : array-like
            The input data for the variable.
        """
        super().__init__()
        self.data = da.array(data)

    def __getitem__(self, key):
        """
        Retrieve data by key.

        Parameters
        ----------
        key : Any
            The key for retrieving data.

        Returns
        -------
        dask.array.Array
            The corresponding data for the given key.
        """
        return self.data[key]
    
    def compute(self):
        """
        Compute the dask array, returning a concrete result.
        
        Returns
        -------
        numpy.ndarray
            The computed result.
        """
        if isinstance(self.data, da.Array):
            return self.data.compute()

class Instants(CustomAttributes):
    """
    Class to manage variables within an instant of a zone.

    Attributes
    ----------
    variables : OrderedDict
        Stores the variables for the instant.
    """
    def __init__(self):
        """
        Initialize the Instants object with an empty set of variables.
        """
        super().__init__()
        self.variables = OrderedDict() 

    def add_variable(self, name, data):
        """
        Add or update a variable in the instant.

        If the variable already exists, the data is updated, otherwise, a new variable is created.

        Parameters
        ----------
        name : str
            The name of the variable.
        data : array-like
            The data associated with the variable, converted to a Dask array.
        """
        if name in self.variables:
        # This line updates existing variable, not overwrite
            self.variables[name].data = da.array(data)
        else:
            self.variables[name] = Variables(data)

    def delete_variable(self, names):
        """
        Delete one or more variables from the instant.

        Parameters
        ----------
        names : str or list of str
            The name(s) of the variables to delete.
        """
        if not isinstance(names, list):
            names = [names]
        for name in names:
            self.variables.pop(name, None)
    
    def rename_variable(self, old_names, new_names):
        """
        Rename one or more variables in the instant.

        Parameters
        ----------
        old_names : str or list of str
            The current name(s) of the variables to rename.
        new_names : str or list of str
            The new name(s) to assign to the variables.

        Raises
        ------
        ValueError
            If the number of old and new names doesn't match, or if an old name doesn't exist.
        """
        if not isinstance(old_names, list):
            old_names = [old_names]
        if not isinstance(new_names, list):
            new_names = [new_names]
        if len(old_names) != len(new_names):
            raise ValueError("The number of old and new variable names must match.")
        
        for new_name, old_name in zip(new_names, old_names):
            if old_name in self.variables:
                self.variables[new_name] = self.variables.pop(old_name)
            else:
                raise ValueError(f"{old_name} not found")

    def __getitem__(self, key):
        """
        Access a variable by name or index.

        Parameters
        ----------
        key : int or str
            The index or name of the variable.

        Returns
        -------
        Variables
            The variable associated with the given key.
        """
        if isinstance(key, int):
            return list(self.variables.values())[key]
        return self.variables[key]
    
    def items(self):
        """
        Iterate over the variables and their data in the instant.

        Yields
        ------
        tuple
            A tuple containing the variable name and its associated data.
        """
        for variable_name, variable_obj in self.variables.items():
            yield (variable_name, variable_obj.data)
        
    def keys(self):
        """
        Return the keys (names) of all variables in the instant.

        Returns
        -------
        KeysView
            A view object that displays the variable names.
        """
        return self.variables.keys()
    
    def compute(self, expression, verbose=DEFAULT_VERBOSE):
        """
        Compute a new variable based on the given expression.

        Parameters
        ----------
        expression : str
            The expression to compute. For example, "variable3 = variable1 * variable2".
        verbose : bool, optional
            If True, display progress bar. Default is False.

        Example
        -------
        >>> instant.compute('variable3 = variable1 * variable2', verbose=True)
        >>> instant.compute('Time = 5 + Time')  # Handles self-referential computations
        """
        if verbose: self.print_text("info", "\nComputing")
        var_name, operation = expression.split('=', 1)
        var_name = var_name.strip()
        operation = operation.strip()

        variables_in_expression = set(self.variables.keys())
        variables_used = set(var for var in variables_in_expression 
                             if bool(re.search(rf'\b{re.escape(var)}\b', operation)))

        if all(var in self.variables for var in variables_used):
            local_namespace = {var: self.variables[var].data for var in variables_used}
            
            try:
                for _ in self.tqdm_wrapper(range(1), desc="Variables", verbose=verbose):
                    result = eval(operation, globals(), local_namespace)
                    
                    if var_name in self.variables:
                        self.variables[var_name].data = da.array(result)
                    else:
                        self.add_variable(var_name, result)
            except Exception as e:
                print(f"Error evaluating expression '{expression}': {e}")
        else:
            print(f"Not all required variables exist in this instant for expression: {expression}")

class Zones(CustomAttributes):
    """
    Class representing a collection of instants within a zone.

    Attributes
    ----------
    instants : OrderedDict
        Dictionary to store `Instants` objects, where each key is an instant name.
    """
    def __init__(self):
        """
        Initialize the Zones object.
        """
        super().__init__()
        self.instants = OrderedDict() 

    def add_instant(self, names) :
        """
        Add one or more instants to the zone.

        Parameters
        ----------
        names : Union[str, List[str]]
            The name or list of names of instants to add.

        Example
        -------
        >>> zones.add_instant('instant1')
        >>> zones.add_instant(['instant1', 'instant2'])
        """
        if not isinstance(names, list):
            names = [names]
            
        for name in names:
            self.instants[name] = Instants()

    def delete_instant(self, names):
        """
        Delete one or more instants from the zone.

        Parameters
        ----------
        names : Union[str, List[str]]
            The name or list of names of instants to delete.

        Example
        -------
        >>> zones.delete_instant('instant1')
        >>> zones.delete_instant(['instant1', 'instant2'])
        """
        if not isinstance(names, list):
            names = [names]
            
        for name in names:
            self.instants.pop(name, None)

    def rename_instant(self, old_names, new_names):
        """
        Rename one or more instants in the zone.

        Parameters
        ----------
        old_names : Union[str, List[str]]
            The current name or list of current names of instants to rename.
        new_names : Union[str, List[str]]
            The new name or list of new names for the instants.

        Raises
        ------
        ValueError
            If the number of old names and new names does not match.
        ValueError
            If an old name is not found in the instants.

        Example
        -------
        >>> zones.rename_instant('old_instant', 'new_instant')
        >>> zones.rename_instant(['old1', 'old2'], ['new1', 'new2'])
        """
        if not isinstance(old_names, list):
            old_names = [old_names]
        if not isinstance(new_names, list):
            new_names = [new_names]
        if len(old_names) != len(new_names):
            raise ValueError("The number of old and new instant names must match.")
        
        for new_name, old_name in zip(new_names, old_names):
            if old_name in self.instants:
                self.instants[new_name] = self.instants.pop(old_name)
            else:
                raise ValueError(f"{old_name} not found")

    def __getitem__(self, key):
        """
        Get an instant by index or name.

        Parameters
        ----------
        key : Union[int, str]
            The index or name of the instant to retrieve.

        Returns
        -------
        Instants
            The `Instants` object corresponding to the given key.

        Example
        -------
        >>> instant = zones[0]
        >>> instant = zones['instant1']
        """
        if isinstance(key, int):
            return list(self.instants.values())[key]
        return self.instants[key]
    
    def items(self):
        """
        Get an iterator over all instant and variable name pairs.

        Returns
        -------
        Generator[Tuple[str, str], None, None]
            An iterator yielding pairs of instant names and variable names.

        Example
        -------
        >>> for instant_name, var_name in zones.items():
        >>>     print(f"Instant: {instant_name}, Variable: {var_name}")
        """
        for instant_name, instant_obj in self.instants.items():
            for var_name in instant_obj.keys():
                yield (instant_name, var_name)
    
    def keys(self):
        """
        Get an iterator over the names of all instants in the zone.

        Returns
        -------
        Generator[str, None, None]
            An iterator yielding the names of all instants.

        Example
        -------
        >>> for instant_name in zones.keys():
        >>>     print(instant_name)
        """
        return self.instants.keys()
    
    def compute(self, expression, verbose=DEFAULT_VERBOSE):
        """
        Compute a new variable based on the given expression and apply it across all instants in the zone.

        Parameters
        ----------
        expression : str
            The expression to compute. For example, "variable3 = variable1 * variable2".
        verbose : bool, optional
            If True, display progress bar. Default is False.

        Example
        -------
        >>> zone.compute('variable3 = variable1 * variable2', verbose=True)
        """
        if verbose: self.print_text("info", "\nComputing")
        for instant in self.tqdm_wrapper(self.instants.values(), desc="Instants ", verbose=verbose):
            instant.compute(expression, verbose = False)

class Base(CustomAttributes):
    """
    Base class representing a structure containing zones and instants.

    Attributes
    ----------
    zones : OrderedDict
        A dictionary to store `Zones` objects, where each key represents a zone name.
    """
    def __init__(self):
        """
        Initialize the Base object.
        """
        super().__init__()
        self.zones = OrderedDict()

    def init(self, zones = None, instants = None):
        """
        Initialize zones and instants. Adds default zones and instants if none are provided.

        Parameters
        ----------
        zones : Union[str, List[str]], optional
            The name or list of names of zones to initialize. Defaults to `DEFAULT_ZONE_NAME`.
        instants : Union[str, List[str]], optional
            The name or list of names of instants to initialize. Defaults to `DEFAULT_INSTANT_NAME`.

        Example
        -------
        >>> base.init(['zone1'], ['instant1'])
        """
        if zones is None:
            zones = DEFAULT_ZONE_NAME
        
        if instants is None:
            instants = DEFAULT_INSTANT_NAME
            
        for zone in zones:
            self.add_zone(zone)
            for instant in instants:
                self[zone].add_instant(instant)
    
    def add(self, zones = None, instants = None):
        """
        Add zones and instants to the Base object.

        Parameters
        ----------
        zones : Union[str, List[str]], optional
            Names of the zones to add.
        instants : Union[str, List[str]], optional
            Names of the instants to add.

        Example
        -------
        >>> base.add(['zone1'], ['instant1'])
        """
        self.init(zones, instants)

    def add_zone(self, names):
        """
        Add one or more zones to the Base object.

        Parameters
        ----------
        names : Union[str, List[str]]
            The name or list of names of zones to add.

        Example
        -------
        >>> base.add_zone('zone1')
        >>> base.add_zone(['zone1', 'zone2'])
        """
        if not isinstance(names, list):
            names = [names]
            
        for name in names:
            self.zones[name] = Zones()

    def delete_zone(self, names):
        """
        Delete one or more zones from the Base object.

        Parameters
        ----------
        names : Union[str, List[str]]
            The name or list of names of zones to delete.

        Example
        -------
        >>> base.delete_zone('zone1')
        >>> base.delete_zone(['zone1', 'zone2'])
        """
        if not isinstance(names, list):
            names = [names]
            
        for name in names:
            self.zones.pop(name, None) 

    def rename_zone(self, old_names, new_names):
        """
        Rename one or more zones in the Base object.

        Parameters
        ----------
        old_names : Union[str, List[str]]
            The current name or list of names of zones to rename.
        new_names : Union[str, List[str]]
            The new name or list of names for the zones.

        Raises
        ------
        ValueError
            If the number of old and new zone names does not match.
        ValueError
            If a zone with the old name is not found.

        Example
        -------
        >>> base.rename_zone('old_zone', 'new_zone')
        >>> base.rename_zone(['old1', 'old2'], ['new1', 'new2'])
        """
        if not isinstance(old_names, list):
            old_names = [old_names]
        if not isinstance(new_names, list):
            new_names = [new_names]
            
        if len(old_names) != len(new_names):
            raise ValueError("The number of old and new zone names must match.")
        
        for new_name, old_name in zip(new_names, old_names):
            if old_name in self.zones:
                self.zones[new_name] = self.zones.pop(old_name)
            else:
                raise ValueError(f"{old_name} not found")
    
    def rename_variable(self, old_names, new_names):
        """
        Rename one or more variables across all zones and instants.

        Parameters
        ----------
        old_names : List[str]
            The list of current names of variables to rename.
        new_names : List[str]
            The list of new names for the variables.

        Raises
        ------
        ValueError
            If the number of old and new variable names does not match.

        Example
        -------
        >>> base.rename_variable(['var1'], ['new_var1'])
        """
        if len(old_names) != len(new_names):
            raise ValueError("The number of old and new variable names must match.")

        # Dictionary to track locations of old variables
        variable_map = defaultdict(list)
        
        # Build the variable map by checking all zones and instants
        for zone_name, zone in self.zones.items():
            for instant_name, instant in zone.instants.items():
                for old_name in old_names:
                    if old_name in instant.variables:
                        variable_map[old_name].append((zone_name, instant_name))

        # Now rename the variables in the locations found
        for old_name, new_name in zip(old_names, new_names):
            if old_name in variable_map:
                # Go through all zones and instants where the variable exists
                for zone_name, instant_name in variable_map[old_name]:
                    instant = self.zones[zone_name][instant_name]
                    instant.rename_variable(old_name, new_name)

    def __getitem__(self, key):
        """
        Get a zone by index or name.

        Parameters
        ----------
        key : Union[int, str]
            The index or name of the zone to retrieve.

        Returns
        -------
        Zones
            The `Zones` object corresponding to the given key.

        Example
        -------
        >>> zone = base[0]
        >>> zone = base['zone1']
        """
        if isinstance(key, int):
            return list(self.zones.values())[key]
        return self.zones[key]
    
    def items(self):
        """
        Get an iterator over all zone and instant name pairs.

        Returns
        -------
        Generator[Tuple[str, str], None, None]
            An iterator yielding pairs of zone names and instant names.

        Example
        -------
        >>> for zone_name, instant_name in base.items():
        >>>     print(f"Zone: {zone_name}, Instant: {instant_name}")
        """
        for zone_name, zone_obj in self.zones.items():
            for instant_name in zone_obj.keys():
                yield (zone_name, instant_name)
    
    def keys(self):
        """
        Get an iterator over the names of all zones in the Base object.

        Returns
        -------
        Generator[str, None, None]
            An iterator yielding the names of all zones.

        Example
        -------
        >>> for zone_name in base.keys():
        >>>     print(zone_name)
        """
        return self.zones.keys()
    
    def compute(self, expression, verbose=DEFAULT_VERBOSE, chunk_size=200):
        """
        Compute a new variable for all instants based on the provided expression.

        The expression should be of the form 'new_variable = operation', where the 
        operation is a mathematical expression involving existing variables. This 
        method updates the instants in the Base object with the computed values.

        Parameters
        ----------
        expression : str
            A mathematical expression of the form 'new_variable = operation'. The 
            new variable will be computed for all instants.
        verbose : bool, optional
            If True, print progress information (default is DEFAULT_VERBOSE).
        chunk_size : int, optional
            The number of instants to process per chunk in multiprocessing mode. 
            Set to None to disable chunking. Default is 200.

        Returns
        -------
        None

        Notes
        -----
        This method supports both multiprocessing and multithreading to parallelize 
        the computation over multiple instants. When the number of chunks is greater 
        than the chunk size, multiprocessing is used; otherwise, multithreading is 
        employed.

        The results of the computation are stored in the Base object by updating the 
        corresponding variables for each instant. If a variable does not exist, it 
        will be created.

        Examples
        --------
        >>> base.compute('new_var = var1 + var2', verbose=True)
        """
        if verbose: self.print_text("info", "\nComputing")
        
        var_name, operation = expression.split('=', 1)
        var_name = var_name.strip()
        operation = operation.strip()

        variables_in_expression = set()
        for zone in self.zones.values():
            for instant in zone.instants.values():
                variables_in_expression.update(instant.variables.keys())

        variables_used = set(var for var in variables_in_expression 
                            if bool(re.search(rf'\b{re.escape(var)}\b', operation)))

        all_instants = [
            (zone_id, instant_id, {var: instant.variables[var].data for var in instant.variables})
            for zone_id, zone in self.zones.items()
            for instant_id, instant in zone.instants.items()
        ]
        
        if chunk_size is None: 
            chunk_size = 0
        else:
            chunks = [all_instants[i:i + chunk_size] for i in range(0, len(all_instants), chunk_size)]
        
        # Use multiprocessing
        if len(chunks) >= 10:
            print("Multiprocess computation")
            cpu_count = multiprocessing.cpu_count()
            with ProcessPoolExecutor(max_workers=cpu_count) as executor:
                futures = [executor.submit(self.process_chunk, chunk, var_name, operation, variables_used) for chunk in chunks]

                all_results = []
                total = len(futures)
                with self.tqdm_wrapper(total, desc="Progress", verbose=verbose) as pbar:
                    for future in as_completed(futures):
                        all_results.extend(future.result())
                        pbar.update()

            # Update the Base object with the computed results
            for zone_id, instant_id, var_name, result in all_results:
                if result is not None:
                    instant = self.zones[zone_id].instants[instant_id]
                    if var_name in instant.variables:
                        instant.variables[var_name].data = result
                    else:
                        instant.add_variable(var_name, result)
        
        # Use MultiThreading
        else:
            print("Multithread computation")
            def evaluate_add_multithread(instant, var_name, operation, variables_used):
                local_namespace = {var: instant.variables[var].data for var in variables_used if var in instant.variables}
                
                try:
                    result = eval(operation, globals(), local_namespace)
                    if var_name in instant.variables:
                        instant.variables[var_name].data = da.array(result)
                    else:
                        instant.add_variable(var_name, result)
                except Exception as e:
                    print(f"Error evaluating expression '{operation}' in instant: {e}")

            with ThreadPoolExecutor() as executor:
                all_results = []
                for zone in self.zones.values():
                    for instant in zone.instants.values():
                        if not variables_used or all(var in instant.variables for var in variables_used):
                            future = executor.submit(evaluate_add_multithread,
                                                    instant, var_name, operation, variables_used)
                            all_results.append(future)

                # Wait for all computations to finish
                for future in self.tqdm_wrapper(all_results, desc="Zones    ", 
                                        verbose=verbose):
                    future.result()

    def show(self, stats = False):
        """
        Display information about the Base object, including zone, instant, and variable details.

        Parameters
        ----------
        stats : bool, optional
            If True, display statistics (min, mean, max) for each variable. Default is False.

        Example
        -------
        >>> base.show()
        >>> base.show(stats=True)
        """
        self.print_text("text", "\nBase: ")
        for zone_name, zone in self.zones.items():
            self.print_text("check", f"  Zone: {zone_name}")
            for instant_name, instant in zone.instants.items():
                self.print_text("check", f"    Instant: {instant_name}")
                for var_name, variable in instant.variables.items():
                    data = variable.data
                    msg = \
                        f"      Variable: {var_name} -> " + \
                        f"Shape :{data.shape}"
                    if stats:
                        msg += \
                            f", stats(min, mean, max): " + \
                            f"({np.round(data.min().compute(), 2)}, " + \
                            f"{np.round(data.mean().compute(), 2)}, " + \
                            f"{np.round(data.max().compute(), 2)})"
                    self.print_text("blue", msg)
    
    @staticmethod
    def evaluate_multiprocess(instant_data, operation, variables_used):
        """
        Evaluate a mathematical expression in a multiprocessing context for a single instant.

        Parameters
        ----------
        instant_data : dict
            A dictionary containing variable data for a given instant. Keys are variable 
            names, and values are the corresponding data arrays.
        operation : str
            A mathematical expression to evaluate. The expression can include variables 
            present in the `instant_data` dictionary.
        variables_used : set
            A set of variable names that are used in the expression.

        Returns
        -------
        dask.array.Array or None
            The result of the evaluated expression as a Dask array. Returns None if 
            an error occurs during evaluation.

        Notes
        -----
        This method uses the `eval` function to dynamically evaluate the mathematical 
        expression within a local namespace constructed from the variables present 
        in `instant_data`. If an error occurs during evaluation, the method catches 
        the exception and prints an error message.

        Example
        -------
        >>> result = Base.evaluate_multiprocess(instant_data, 'var1 + var2', {'var1', 'var2'})
        >>> if result is not None:
        >>>     print("Computation succeeded")
        """
        local_namespace = {var: instant_data[var] for var in variables_used if var in instant_data}
        
        try:
            result = eval(operation, globals(), local_namespace)
            return da.array(result)
        except Exception as e:
            print(f"Error evaluating expression '{operation}' in instant: {e}")
            return None

    @staticmethod
    def process_chunk(chunk, var_name, operation, variables_used):
        """
        Process a chunk of instants by evaluating a mathematical expression for each instant.

        Parameters
        ----------
        chunk : list of tuples
            A list of tuples, where each tuple contains:
            - zone_id: The ID of the zone.
            - instant_id: The ID of the instant.
            - instant_data: A dictionary of variable data for the instant.
        var_name : str
            The name of the variable to store the result of the evaluated expression.
        operation : str
            A mathematical expression to evaluate.
        variables_used : set
            A set of variable names that are used in the expression.

        Returns
        -------
        list of tuples
            A list of tuples, where each tuple contains:
            - zone_id: The ID of the zone.
            - instant_id: The ID of the instant.
            - var_name: The name of the variable to be updated.
            - result: The computed result as a Dask array or None if an error occurred.

        Notes
        -----
        This method iterates over all instants in the chunk, evaluates the expression 
        for each instant using `evaluate_multiprocess`, and returns the results.

        Example
        -------
        >>> chunk = [(zone_id1, instant_id1, instant_data1), (zone_id2, instant_id2, instant_data2)]
        >>> results = Base.process_chunk(chunk, 'new_var', 'var1 + var2', {'var1', 'var2'})
        >>> for zone_id, instant_id, var_name, result in results:
        >>>     print(f"Zone {zone_id}, Instant {instant_id}: {var_name} = {result}")
        """
        results = []
        for zone_id, instant_id, instant_data in chunk:
            if not variables_used or all(var in instant_data for var in variables_used):
                result = Base.evaluate_multiprocess(instant_data, operation, variables_used)
                results.append((zone_id, instant_id, var_name, result))
        return results

if __name__ == "__main__":
    import time
    start_time = time.time()
    # Example usage 1
    base = Base()
    
    base.add_zone("Zone1")
    base.add_zone("Zone2")
    base["Zone1"].add_instant("Instant1")
    base["Zone2"].add_instant("Instant1")
    base["Zone1"]["Instant1"].add_variable("Velocity", np.random.randint(1, 101, size=(int(8), 30)))
    base["Zone1"]["Instant1"].add_variable("Pressure", np.random.randint(1, 101, size=(int(8), 30)))
    base["Zone2"]["Instant1"].add_variable("Velocity", np.random.randint(1, 101, size=(int(8), 30)))
    base["Zone2"]["Instant1"].add_variable("Pressure", np.random.randint(1, 101, size=(int(8), 30)))
    base["Zone2"]["Instant1"].add_variable("Density", np.random.randint(1, 101, size=(int(8), 30)))

    # Set custom attributes at different levels
    base.set_attribute('project_name', 'CGNS Project')
    base["Zone1"].set_attribute('zone_type', 'Flow')
    base["Zone1"]["Instant1"].set_attribute('instant_time', '2024-08-28')
    base["Zone1"]["Instant1"]["Velocity"].set_attribute('description', 'Velocity data')
    
    # Accessing custom attributes
    print(base.get_attribute('project_name'))  # Should print 'CGNS Project'
    print(base["Zone1"].get_attribute('zone_type'))  # Should print 'Flow'
    print(base["Zone1"]["Instant1"].get_attribute('instant_time'))  # Should print '2024-08-28'
    print(base["Zone1"]["Instant1"]["Velocity"].get_attribute('description'))  # Should print 'Velocity data'

    # Accessing the default data attribute
    print(base["Zone1"]["Instant1"]["Velocity"].data)  # Should print the data array


    # Example usage 2
    base2 = Base()
    base2.init(zones=["Zone1", "Zone2"], instants=["Instant1", "Instant2"])
    
    base3 = Base()
    base3.init(zones=["Zone1", "Zone2"], instants=["Instant1", "Instant2"])

    # Add some variables to the instants
    npoint = 50e5
    base3["Zone1"]["Instant1"].add_variable("varia1", np.random.randint(1, 101, size=(int(npoint), 30)))
    base3["Zone1"]["Instant1"].add_variable("variable2", np.random.randint(1, 101, size=(int(npoint), 30)))
    base3["Zone1"]["Instant1"].add_variable("var_float", np.random.uniform(low=0.0, high=1.0, size=(int(npoint), 30)))
    
    base3["Zone2"]["Instant2"].add_variable("varle1", np.random.randint(1, 101, size=(int(npoint), 30)))
    base3["Zone2"]["Instant2"].add_variable("variable2", np.random.randint(1, 101, size=(int(npoint), 30)))

    # Compute new variables
    base3.compute("variable3 = varia1 * variable2")  # Computed only where both variable1 and variable2 exist
    base3.compute("variable4 = variable2 * 3")  # Computed in all instants since no variables were missing

    # Check the results
    print("Zone1, Instant1, Variable3:", base3["Zone1"]["Instant1"]["variable3"].data)  
    print("Zone1, Instant1, Variable4:", base3["Zone1"]["Instant1"]["variable4"].data)  
    # print("Zone2, Instant2, Variable3:", base3["Zone2"]["Instant2"]["variable3"].data)
    
    # End the timer
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time} seconds")