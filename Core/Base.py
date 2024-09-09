import numpy as np
import re
import dask.array as da
from typing import Union, Generator, Dict, List, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict, defaultdict

# Orion parameters
from Core import DEFAULT_ZONE_NAME, DEFAULT_INSTANT_NAME

class CustomAttributes:
    def __init__(self):
        self._attributes: Dict[str, Any] = OrderedDict()

    def set_attribute(self, name: str, value: Any) -> None:
        self._attributes[name] = value

    def get_attribute(self, name: str, default: Any = None) -> Any:
        return self._attributes.get(name, default)

    def delete_attribute(self, name: str) -> None:
        self._attributes.pop(name, None)

    def rename_attribute(self, old_name: str, new_name: str) -> None:
        if old_name in self._attributes:
            self._attributes[new_name] = self._attributes.pop(old_name)

class Variables(CustomAttributes):
    def __init__(self, data):
        super().__init__()
        self.data = da.array(data)

    def __getitem__(self, key: Any):
        return self.data[key]
    
    def compute(self):
        if isinstance(self.data, da.Array):
            return self.data.compute()

class Instants(CustomAttributes):
    def __init__(self):
        super().__init__()
        self.variables: Dict[str, Variables] = OrderedDict() 

    def add_variable(self, name, data):
        self.variables[name] = Variables(data)

    def delete_variable(self, names):
        if not isinstance(names, list):
            names = [names]
            
        for name in names:
            self.variables.pop(name, None)
    
    def rename_variable(self, old_names, new_names):
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

    def __getitem__(self, key: Union[int, str]) -> Variables:
        if isinstance(key, int):
            return list(self.variables.values())[key]
        return self.variables[key]

    def compute_variable(self, expression, variable_name):
        # Lazy computation of a variable
        if variable_name not in self.variables:
            exec(expression, globals(), locals())
            self.add_variable(variable_name, locals()[variable_name])
            self[variable_name].compute()
    
    def items(self):
        for variable_name, variable_obj in self.variables.items():
            yield (variable_name, variable_obj.data)
        
    def keys(self):
        return self.variables.keys()
    

class Zones(CustomAttributes):
    def __init__(self):
        super().__init__()
        self.instants: Dict[str, Instants] = OrderedDict() 

    def add_instant(self, names) :
        if not isinstance(names, list):
            names = [names]
            
        for name in names:
            self.instants[name] = Instants()

    def delete_instant(self, names):
        if not isinstance(names, list):
            names = [names]
            
        for name in names:
            self.instants.pop(name, None)

    def rename_instant(self, old_names, new_names) -> None:
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

    def __getitem__(self, key: Union[int, str]) -> Instants:
        if isinstance(key, int):
            return list(self.instants.values())[key]
        return self.instants[key]
    
    def items(self):
        for instant_name, instant_obj in self.instants.items():
            for var_name in instant_obj.keys():
                yield (instant_name, var_name)
    
    def keys(self):
        return self.instants.keys()

class Base(CustomAttributes):
    def __init__(self):
        super().__init__()
        self.zones: Dict[str, Zones] = OrderedDict()

    def init(self, zones: List[str] = None, instants: List[str] = None) -> None:
                
        if zones is None:
            zones = DEFAULT_ZONE_NAME
        
        if instants is None:
            instants = DEFAULT_INSTANT_NAME
            
        for zone in zones:
            self.add_zone(zone)
            for instant in instants:
                self[zone].add_instant(instant)
    
    def add(self, zones: List[str] = None, instants: List[str] = None) -> None:
        self.init(zones, instants)

    def add_zone(self, names):
        if not isinstance(names, list):
            names = [names]
            
        for name in names:
            self.zones[name] = Zones()

    def delete_zone(self, names):
        if not isinstance(names, list):
            names = [names]
            
        for name in names:
            self.zones.pop(name, None) 

    def rename_zone(self, old_names, new_names) -> None:
        
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
    
    def rename_variable(self, old_names, new_names) -> None:
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
        if isinstance(key, int):
            return list(self.zones.values())[key]
        return self.zones[key]
    
    def items(self):
        for zone_name, zone_obj in self.zones.items():
            for instant_name in zone_obj.keys():
                yield (zone_name, instant_name)
    
    def keys(self):
        return self.zones.keys()
    
    def compute(self, expression):
        """Compute a new variable based on the given expression and apply it across all zones and instants."""
        # Parse the expression: "variable3 = variable1 * variable2"
        var_name, operation = expression.split('=', 1)
        var_name = var_name.strip()
        operation = operation.strip()

        # Identify all variables mentioned in the expression
        variables_in_expression = set()
        for zone in self.zones.values():
            for instant in zone.instants.values():
                variables_in_expression.update(instant.variables.keys())

        variables_used = set()
        for var in variables_in_expression:
            if bool(re.search(rf'\b{re.escape(var)}\b', operation)):
                variables_used.add(var)

        # Function to evaluate and add the computed variable
        def evaluate_and_add_variable(instant, var_name, local_operation):
            if var_name in instant.variables:
                instant.delete_variable(var_name)

            try:
                result = eval(local_operation)
                instant.add_variable(var_name, result)
            except Exception as e:
                print(f"Error evaluating expression '{operation}': {e}")

        with ThreadPoolExecutor() as executor:
            futures = []
            if not variables_used:
                # If no variables are used in the expression, compute in all instants
                for zone in self.zones.values():
                    for instant in zone.instants.values():
                        futures.append(
                            executor.submit(evaluate_and_add_variable, instant, var_name, operation)
                        )
            else:
                # Compute only in instants where all variables in the expression exist
                for zone in self.zones.values():
                    for instant in zone.instants.values():
                        if all(var in instant.variables for var in variables_used):
                            # Replace variables in the operation with their corresponding data
                            local_operation = operation
                            for var in variables_used:
                                local_operation = local_operation.replace(var, f'instant.variables["{var}"].data')

                            futures.append(
                                executor.submit(evaluate_and_add_variable, instant, var_name, local_operation)
                            )

            # Wait for all computations to finish
            for future in futures:
                future.result()

    def show(self, stat = False):
        print("Base")
        for zone_name, zone in self.zones.items():
            print(f"  Zone: {zone_name}")
            for instant_name, instant in zone.instants.items():
                print(f"    Instant: {instant_name}")
                for var_name, variable in instant.variables.items():
                    data = variable.data
                    msg = \
                        f"      Variable: {var_name} -> " + \
                        f"Shape :{data.shape}"
                    if stat:
                        msg += \
                            f", stats(min, mean, max): " + \
                            f"({np.round(data.min().compute(), 2)}, " + \
                            f"{np.round(data.mean().compute(), 2)}, " + \
                            f"{np.round(data.max().compute(), 2)})"
                    print(msg)

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