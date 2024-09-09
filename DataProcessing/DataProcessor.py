# Path and Char manipulation
import os
import sys
import pathlib
import re

# Tools
import copy
import itertools
import logging
from collections import deque

# Data processing
import numpy as np
import dask.array as da
import scipy as spy
import pandas as pd

# I/O
import io
import csv
import h5py as hdf
import ast

# Signal processing
from scipy.signal import butter, cheby1, cheby2, ellip, bessel, filtfilt
from scipy import signal
from scipy.optimize import minimize

# Orion
import Core as Orion
from SharedMethods import SharedMethods

# Message mode
import warnings
Verbose = Orion.DEFAULT_VERBOSE
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['VTK_SILENCE_GET_VOID_POINTER_WARNINGS'] = '1'

class Processor(SharedMethods):
    """
    Data processing 
    """
    def __init__(self, base, quantities_of_interest = None,
                 NumericalScheme = None, gridFields = None):
        """
        Initialize the ExportDataToWrite subclass.

        :param base: The dataset containing the data to be exported.
        :type base: Dataset
        :param quantities_of_interest_dic: Dictionary specifying quantities of interest (default is None).
        :type quantities_of_interest_dic: dict, optional
        :param NumericalScheme: The numerical scheme used for export (default is None).
        :type NumericalScheme: str, optional
        :param gridFields: Fields related to the grid (default is None).
        :type gridFields: dict, optional

        .. note::
        
            - This method initializes the ExportDataToWrite subclass with provided parameters.
            - It sets attributes for the dataset (`base`), quantities of interest (`quantities_of_interest_dic`),
            numerical scheme (`NumericalScheme`), and grid fields (`gridFields`).
        """
        super().__init__()
        if quantities_of_interest is None:
            self.base = base
        elif isinstance(base, (list, dict)):
            self.print_text("error", "The reduction of variables currently supports only a single base input at a time.\n")
            sys.exit()
        else:
            self.quantities_of_interest = quantities_of_interest
            self.base = self.__reudce_variables(base, quantities_of_interest)

        self.NumericalScheme = NumericalScheme
        self.gridFields = gridFields
        
    def fusion(self):
        """
        Merge multiple data sources into a single pivot base.

        :param self: The instance of the class that this method belongs to.
        :type self: YourClassName

        :return: A merged data structure containing all data from the base sources.
        :rtype: PivotBase

        .. note::

            - This method initializes the merge with the first data source as the pivot base.
            - It iterates through the remaining data sources, updating the pivot base with zones, instants, and variables.
            - If a zone or instant is missing in the pivot base, it is added.
            - Duplicate variables within the same zone and instant are detected and renamed to avoid conflicts, with a warning message printed.
            - Ensure that variable names are unique to prevent unexpected behavior.
        """
        bases = deque(self.base)
        pivot_base = bases.popleft()
        
        for base in bases:
            for n, (zone, instant) in enumerate(base.items()):

                if zone not in pivot_base.keys():
                    if n == 0: pivot_base.add_zone(zone)
                    pivot_base[zone].add_instant(instant)
                    for var in base[zone][instant].keys():
                        pivot_base[zone][instant].add_variable(var,
                                                            base[zone][instant][var].data)
                else:
                    if instant not in pivot_base[zone].keys():
                        pivot_base[zone].add_instant(instant)
                        for var in base[zone][instant].keys():
                            pivot_base[zone][instant].add_variable(var,
                                                                base[zone][instant][var].data)
                    else:
                        for var in base[zone][instant].keys():
                            if var not in pivot_base[zone][instant].keys():
                                pivot_base[zone][instant].add_variable(var,
                                                                    base[zone][instant][var].data)
                            else:
                                self.print_text("warning", f"Warning duplicate variables detected : {var} renamed to {var}_DUP")
                                pivot_base[zone][instant].add_variable(f"{var}_DUP",
                                                                    base[zone][instant][var].data)
        return pivot_base

    def fft(self, dt, decomposition_type="both", frequencies_band = (None, None), **kwargs):
        """
        Compute the Fast Fourier Transform (FFT) of the input signal for each variable in the dataset.

        :param dt: The time interval between samples in the input signal.
        :type dt: float
        :param decomposition_type: Specifies the type of decomposition to return. Options are:
            - "im/re": Returns the real and imaginary components.
            - "mod/phi": Returns the magnitude and phase components.
            - "both": Returns both real/imaginary and magnitude/phase components (default).
        :type decomposition_type: str, optional
        :param frequencies_band: A tuple or list defining the frequency band to apply for filtering.
            - If `None`, no frequency filtering is applied (default: (None, None)).
            - Values must be in the form (min_freq, max_freq). Frequencies outside this range are excluded.
        :type frequencies_band: tuple or list of float, optional
        :param kwargs: Additional arguments. This can include:
            - `invariant_variables`: A list of variable names to exclude from FFT computation.
            Default is set to Orion.DEFAULT_COORDINATE_NAMES + Orion.DEFAULT_TIME_NAME.
            
        :return: A dictionary-like object (fft_base) with the FFT results, varying by `decomposition_type`:
            - For "im/re":
                - **frequencies**: The positive frequencies corresponding to the FFT result.
                - **real**: The real part of the FFT result for each variable.
                - **imaginary**: The imaginary part of the FFT result for each variable.
            - For "mod/phi":
                - **frequencies**: The positive frequencies corresponding to the FFT result.
                - **magnitude**: The magnitude of the FFT result for each variable.
                - **phase**: The phase of the FFT result for each variable.
            - For "both":
                - **frequencies**: The positive frequencies corresponding to the FFT result.
                - **real**: The real part of the FFT result for each variable.
                - **imaginary**: The imaginary part of the FFT result for each variable.
                - **magnitude**: The magnitude of the FFT result for each variable.
                - **phase**: The phase of the FFT result for each variable.
        :rtype: dict-like (fft_base)

        :raises ValueError: 
            - If `frequencies_band` is not a tuple or list, or if it does not contain exactly two elements.
            - If `decomposition_type` is not one of the accepted values ("im/re", "mod/phi", "both").
        :raises TypeError: If `frequencies_band` is not a tuple or list.

        .. note::
            - This function leverages Dask for parallel and out-of-core computation to handle large datasets.
            - Only positive frequencies are considered due to the symmetry of the FFT.
            - The `mask_band` method is used to filter out frequencies outside the specified `frequencies_band`.
            - `fft_base` contains FFT results for each variable that is not listed in `invariant_variables`.
            """
        invariant_variables = kwargs.get("invariant_variables",
                                         Orion.DEFAULT_TIME_NAME)
        
        if np.any(frequencies_band):
            if not isinstance(frequencies_band, (tuple, list)):
                print("Frequencies band type can be tuple or list.")
                raise ValueError
            
            if not isinstance(frequencies_band, (tuple, list)):
                raise TypeError("Frequencies band must be a tuple or list.")
            
            if len(frequencies_band) != 2:
                frequencies_band = [np.min(frequencies_band), np.max(frequencies_band)]
        
        if decomposition_type not in ["im/re", "mod/phi", "both"]:
            raise ValueError("Invalid decomposition_type. Choose from 'im/re', 'mod/phi', or 'both'.")
                
        fft_base = Orion.Base()
        fft_base.add_zone(list(self.base.keys()))
        for ninstant, (zone, instant) in enumerate(self.base.items()):
            fft_base[zone].add_instant(instant)
            for variable, value in self.base[zone][instant].items():
                if variable not in invariant_variables:
                    # Compute the FFT using Dask
                    fft_value = da.fft.fft(value, axis=0)
                    fft_freqs = da.fft.fftfreq(len(value), d=dt)
                    # Only take the positive frequencies (since FFT is symmetric)
                    if np.any(frequencies_band):
                        mask = self.mask_band(fft_freqs, frequencies_band)
                        fft_value = fft_value[:len(fft_value)//2][mask.compute()]
                        fft_freqs = fft_freqs[:len(fft_freqs)//2][mask.compute()]
                    else:
                        fft_value = fft_value[:len(fft_value)//2]
                        fft_freqs = fft_freqs[:len(fft_freqs)//2]
                    
                    if ninstant == 0:
                        fft_base[zone][instant].add_variable('Frequency', fft_freqs)
                    
                    if decomposition_type in ["im/re", "both"]:
                        fft_base[zone][instant].add_variable(f"{variable}_real", 
                                                             np.real(fft_value))
                        fft_base[zone][instant].add_variable(f"{variable}_img", 
                                                             np.imag(fft_value))
                    
                    if decomposition_type in ["mod/phi", "both"]:
                        fft_base[zone][instant].add_variable(f"{variable}_mag", 
                                                             da.abs(fft_value))
                        fft_base[zone][instant].add_variable(f"{variable}_phase", 
                                                             da.angle(fft_value))

        return fft_base

    def filter(self, **kwargs):
        """
        Apply a digital filter to the data using various filter types and configurations.

        :param filter_type: The type of filter to apply. Options include:
            - 'butterworth': Butterworth filter.
            - 'cheby1': Chebyshev Type I filter.
            - 'cheby2': Chebyshev Type II filter.
            - 'elliptic': Elliptic filter.
            - 'bessel': Bessel filter.
        Default is 'butterworth'.
        :type filter_type: str, optional
        
        :param cutoff: The cutoff frequency or frequencies for the filter:
            - For 'low' and 'high' filters, a single cutoff frequency.
            - For 'band' and 'stop' filters, a list or tuple of two cutoff frequencies.
        If `None`, no filtering is applied. Default is `None`.
        :type cutoff: float, list or tuple, optional
        
        :param sampling_rate: The rate at which samples were taken (samples per second). Default is 1000 Hz.
        :type sampling_rate: float, optional
        
        :param order: The order of the filter. Higher values mean a steeper roll-off. Default is 5.
        :type order: int, optional
        
        :param btype: The type of filter to design:
            - 'low': Low-pass filter. Allows frequencies below the cutoff to pass through.
            - 'high': High-pass filter. Allows frequencies above the cutoff to pass through.
            - 'band': Band-pass filter. Allows frequencies within the cutoff range to pass through.
            - 'stop': Band-stop (notch) filter. Blocks frequencies within the cutoff range.
        Default is 'low'.
        :type btype: str, optional

        :return: The modified data with the filter applied.
        :rtype: dict-like (self.base)

        :raises ValueError: 
            - If `cutoff` is not appropriate for the specified `btype`.
            - If `cutoff` frequencies exceed the Nyquist frequency (half the sampling rate).
            - If an unsupported `filter_type` is specified.
        
        .. note::
            - The Nyquist frequency is half the sampling rate, and it is used to normalize the cutoff frequency.
            - The `filtfilt` function is used for zero-phase filtering, ensuring no phase distortion.
        """

        filter_type = kwargs.get('filter_type', 'butterworth')
        cutoff = kwargs.get('cutoff', None)
        sampling_rate = kwargs.get('sampling_rate', 1e3)
        order = kwargs.get('order', 5)
        btype = kwargs.get('btype', 'low')

        # Ensure cutoff is a list if btype is 'band' or 'stop'
        if btype in ['band', 'stop'] and not isinstance(cutoff, (list, tuple)):
            raise ValueError(f"For 'btype'='{btype}', 'cutoff' must be a list or tuple of two values.")
        elif btype not in ['band', 'stop'] and isinstance(cutoff, (list, tuple)):
            raise ValueError(f"For 'btype'='{btype}', 'cutoff' should not be a list or tuple.")

        nyquist_frequency = sampling_rate / 2.0
        if btype in ['band', 'stop']:
            normal_cutoff = [freq / nyquist_frequency for freq in cutoff]
        else:
            normal_cutoff = cutoff / nyquist_frequency

        if (btype in ['band', 'stop'] and (normal_cutoff[0] >= 1 or normal_cutoff[1] >= 1)) or (btype not in ['band', 'stop'] and normal_cutoff >= 1):
            raise ValueError("Cutoff frequency must be less than the Nyquist frequency.")

        if filter_type == 'butterworth':
            b, a = butter(order, normal_cutoff, btype=btype, analog=False)
        elif filter_type == 'cheby1':
            b, a = cheby1(order, 0.5, normal_cutoff, btype=btype, analog=False)
        elif filter_type == 'cheby2':
            b, a = cheby2(order, 20, normal_cutoff, btype=btype, analog=False)
        elif filter_type == 'elliptic':
            b, a = ellip(order, 0.5, 20, normal_cutoff, btype=btype, analog=False)
        elif filter_type == 'bessel':
            b, a = bessel(order, normal_cutoff, btype=btype, analog=False, norm='phase')
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")

        for zone, instant in self.base.items():
            for variable_name, variable_obj in self.base[zone][instant].items():
                self.base[zone][instant].add_variable(variable_name, 
                                                      filtfilt(b, a, 
                                                               variable_obj)
                                                      )
                    
        return self.base
    
    def __is_dask_array(self, input):
        if not isinstance(input, da.Array):
            return da.from_array(input, chunks=(len(input),))
        return input
    
    def __rechunk_array(self, input):
        input = self.__is_dask_array(input)
        if input.chunks[0][0] != len(input):
            return input.rechunk((len(input),))
        return input

    @staticmethod
    def compute_transfer_function(input_signal, output_signal, time, order = (1, 2),
                              method = 'scipy_minimize', initial_params = None,
                              freq_range = None):
        """
        Compute the transfer function between input and output signals.

        Parameters:
        - input_signal: array-like, the input force signal.
        - output_signal: array-like, the output displacement signal.
        - time: array-like, time vector corresponding to the signals.
        - order: tuple of (numerator_order, denominator_order) for the transfer function.
        - method: str, method to use for system identification ('scipy_minimize', 'arx', 'prony', 'pem').
        - initial_params: array-like, initial guess for coefficients. If None, defaults to ones.
        - freq_range: tuple of (min_freq, max_freq) for frequency response. If None, uses default range.

        Returns:
        - system_optimized: scipy.signal.TransferFunction object, the optimized transfer function.
        - result_dict: dictionary containing Bode plot data and additional information.
        """
        input_signal, output_signal, time = map(Processor.prepare_data, [input_signal, output_signal, time])

        # Ensure uniform time intervals
        if np.max(np.diff(time)) != np.min(np.diff(time)):
            time = np.linspace(np.min(time), np.max(time), len(time))
            # input_signal = np.interp(time, Processor.prepare_data(time), input_signal)
            # output_signal = np.interp(time, Processor.prepare_data(time), output_signal)

        num_order, den_order = order
        if initial_params is None:
            initial_params = np.ones(num_order + den_order + 1)

        if method == 'scipy_minimize':
            params_optimized = Processor.fit_scipy_minimize(num_order, den_order, 
                                                            initial_params, 
                                                            input_signal, 
                                                            output_signal, 
                                                            time)
        elif method == 'arx':
            params_optimized = Processor.fit_arx(num_order, den_order, 
                                                 input_signal, output_signal)
        elif method == 'prony':
            params_optimized = Processor.fit_prony(num_order, den_order, 
                                                   output_signal)
        elif method == 'pem':
            params_optimized = Processor.fit_pem(num_order, den_order, 
                                                 input_signal, output_signal)
        else:
            raise ValueError("Unsupported method. Use 'scipy_minimize', 'arx', 'prony', or 'pem'.")

        num_optimized = params_optimized[:num_order + 1]
        den_optimized = np.concatenate(([1.], params_optimized[num_order + 1:]))

        system_optimized = signal.TransferFunction(num_optimized, den_optimized)

        # Check stability
        is_stable = np.all(np.real(np.roots(den_optimized)) < 0)

        # Compute frequency response
        if freq_range:
            w = np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]), 
                            num=1000)
        else:
            w = None

        # Correctly compute Bode plot for continuous-time transfer function
        if system_optimized.dt is None:
            w, mag, phase = signal.bode(system_optimized, w=w)
        else:
            w, mag, phase = signal.dlti(system_optimized).bode(w=w)

        # Compute additional frequency response plots
        w, h = signal.freqs(num_optimized, den_optimized, w)

        result_dict = {
            'frequencies': w,
            'magnitude': mag,
            'phase': phase,
            'is_stable': is_stable,
            'nyquist': h,
            'coherence': np.corrcoef(input_signal, output_signal)[0, 1],
            'params': params_optimized,
            'method': method,
            'order': order
        }

        return system_optimized, result_dict

    @staticmethod
    def mask_band(input_signal, band):
        min_band, max_band = band
        
        # Ensure min_band and max_band are either None or numeric
        if min_band is not None and not isinstance(min_band, (int, float)):
            raise ValueError("min_band must be a numeric value or None.")
        
        if max_band is not None and not isinstance(max_band, (int, float)):
            raise ValueError("max_band must be a numeric value or None.")
        
        # Handle various cases for min_band and max_band
        if min_band is not None and max_band is not None:
            if min_band >= max_band:
                raise ValueError("min_band cannot be greater than max_band.")
            mask = (input_signal >= min_band) & (input_signal <= max_band)
        
        elif min_band is not None and max_band is None:
            mask = input_signal >= min_band
        
        elif min_band is None and max_band is not None:
            mask = input_signal <= max_band
        
        else:
            raise ValueError("At least one of min_band or max_band must be provided.")
        
        return mask
    
    @staticmethod
    def prepare_data(x):
        """ Convert Dask array to NumPy array if necessary. """
        if isinstance(x, da.Array):
            return x.compute()
        return np.asarray(x)

    @staticmethod
    def fit_scipy_minimize(num_order, den_order, initial_params, input_signal, 
                           output_signal, time):
        """ Fit the transfer function using scipy's minimize. """
        def fit_transfer_function(params):
            num = params[:num_order + 1]
            den = np.concatenate(([1.], params[num_order + 1:]))
            system = signal.TransferFunction(num, den)
            _, yout, _ = signal.lsim(system, U=input_signal, T=time)
            return np.sum((yout - output_signal) ** 2)

        result = minimize(fit_transfer_function, initial_params, 
                          method='Nelder-Mead')
        return result.x

    @staticmethod
    def fit_arx(num_order, den_order, input_signal, output_signal):
        """ Fit the ARX model using least squares. """
        from scipy.linalg import lstsq

        num_rows = len(input_signal) - max(num_order, den_order)
        regressor = np.zeros((num_rows, num_order + den_order + 1))

        for i in range(num_order + 1):
            regressor[:, i] = input_signal[max(num_order, den_order) - i : len(input_signal) - i]

        for j in range(1, den_order + 1):
            regressor[:, num_order + j] = -output_signal[max(num_order, den_order) - j : len(output_signal) - j]

        response = output_signal[max(num_order, den_order):]
        params_optimized, _, _, _ = lstsq(regressor, response)

        return params_optimized

    @staticmethod
    def fit_prony(num_order, den_order, output_signal):
        """ Fit the system using the Prony method. """
        N = len(output_signal)
        A = np.vstack([output_signal[i:N-den_order+i] for i in range(den_order)]).T
        b = output_signal[den_order:]
        den_params = np.linalg.lstsq(A, b, rcond=None)[0]
        den_params = np.concatenate(([1.], -den_params))

        # Fix: Compute zeros and poles from den_params
        system_poles = np.roots(den_params)
        system_zeros = np.poly1d(den_params).roots

        # Fit numerator based on the order and poles
        num_params = np.polyfit(system_poles, system_zeros, num_order)
        return np.concatenate((num_params, den_params[1:]))

    @staticmethod
    def fit_pem(num_order, den_order, input_signal, output_signal):
        """ Fit the system using the Prediction Error Method. """
        from scipy.linalg import lstsq

        num_rows = len(input_signal) - max(num_order, den_order)
        regressor = np.zeros((num_rows, num_order + den_order + 1))

        for i in range(num_order + 1):
            regressor[:, i] = input_signal[max(num_order, den_order) - i : len(input_signal) - i]

        for j in range(1, den_order + 1):
            regressor[:, num_order + j] = -output_signal[max(num_order, den_order) - j : len(output_signal) - j]

        response = output_signal[max(num_order, den_order):]
        params_optimized, _, _, _ = lstsq(regressor, response)

        return params_optimized

if __name__ == "__main__":
    # ============================ Case root path ==============================
    # Case Root Path
    path = r"T:\CSprojects\CS-projects\CS13124  CV-RTD valve ext\BHG_Data"
    case_path = SharedMethods().path_manage(path)

    # Inputs Path
    input_common_path = os.path.join(case_path, "EWR13124-876", "G60_Perfo")
    input_name_patterns = [
        "G60-<instant>.MAT"
        ]
    # ============================== Read inputs ===============================
    file_dic = Reader(input_common_path, input_name_patterns).read_mat()


    file_path = os.path.join(input_common_path, list(file_dic.keys())[0])
    file = spy.io.loadmat(file_path)


    # ============================= Data manipulation ==========================